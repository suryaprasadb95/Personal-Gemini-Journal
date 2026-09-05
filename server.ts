import express, { Request, Response } from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = 3000;

// Top-Level Request Deserialization (Ordering Guarantee)
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Lazy initialization of GoogleGenAI client
let genAIClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI {
  if (!genAIClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn('GEMINI_API_KEY environment variable is not set. API calls will fail until configured.');
      throw new Error('GEMINI_API_KEY environment variable is required');
    }
    genAIClient = new GoogleGenAI({ apiKey });
  }
  return genAIClient;
}

// Resilient Model Fallback Ladder
const MODEL_FALLBACK_CHAIN = [
  'gemini-3.6-flash',
  'gemini-3.1-flash-lite',
  'gemini-flash-latest',
  'gemini-3.7-flash'
];

interface FallbackOptions {
  systemInstruction?: string;
  temperature?: number;
}

async function generateContentWithFallback(
  ai: GoogleGenAI,
  contents: any,
  options?: FallbackOptions
): Promise<{ text: string; modelUsed: string }> {
  let lastError: any = null;

  for (const model of MODEL_FALLBACK_CHAIN) {
    try {
      const config: any = {};
      if (options?.systemInstruction) {
        config.systemInstruction = options.systemInstruction;
      }
      if (options?.temperature !== undefined) {
        config.temperature = options.temperature;
      }

      const response = await ai.models.generateContent({
        model,
        contents,
        config: Object.keys(config).length > 0 ? config : undefined,
      });

      if (response && response.text) {
        return { text: response.text, modelUsed: model };
      }
    } catch (err: any) {
      console.warn(`Attempt with model ${model} failed:`, err?.message || err);
      lastError = err;

      const errorMessage = String(err?.message || '');
      const statusCode = err?.status || err?.statusCode || (err?.response && err?.response.status);
      const isRecoverable =
        statusCode === 503 ||
        statusCode === 429 ||
        statusCode === 404 ||
        statusCode === 500 ||
        errorMessage.includes('503') ||
        errorMessage.includes('429') ||
        errorMessage.includes('not found') ||
        errorMessage.includes('overloaded');

      if (!isRecoverable && MODEL_FALLBACK_CHAIN.indexOf(model) === 0) {
        // Continue fallback regardless to provide maximum resilience
      }
    }
  }

  throw new Error(`All Gemini models in fallback ladder failed. Last error: ${lastError?.message || 'Unknown error'}`);
}

// Health check endpoint
app.get('/api/health', (req: Request, res: Response) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// --- RAG (Retrieval-Augmented Generation) Engine & Anti-Hallucination Utilities ---

interface DocumentInput {
  id: string;
  title: string;
  content: string;
  category?: string;
}

interface KnowledgeChunk {
  chunkId: string;
  docId: string;
  docTitle: string;
  category?: string;
  text: string;
  relevanceScore?: number;
}

// Chunks text intelligently by paragraphs or token length
function chunkText(text: string, maxWords = 180, overlapWords = 30): string[] {
  if (!text || !text.trim()) return [];
  const paragraphs = text.split(/\n\s*\n/).map(p => p.trim()).filter(Boolean);
  const chunks: string[] = [];

  for (const para of paragraphs) {
    const words = para.split(/\s+/);
    if (words.length <= maxWords) {
      chunks.push(para);
    } else {
      let startIndex = 0;
      while (startIndex < words.length) {
        const slice = words.slice(startIndex, startIndex + maxWords);
        chunks.push(slice.join(' '));
        startIndex += (maxWords - overlapWords);
      }
    }
  }

  return chunks.length > 0 ? chunks : [text.trim()];
}

// Compute cosine similarity between two numerical vectors
function computeCosineSimilarity(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(vecA.length, vecB.length);
  for (let i = 0; i < len; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Tokenize text for lexical / keyword scoring
const STOP_WORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
  'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'against', 'between', 'into',
  'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
  'of', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
  'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
  'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
  'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'what', 'i', 'my', 'me'
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2 && !STOP_WORDS.has(t));
}

// Lexical BM25-style keyword matching score
function computeLexicalScore(queryTokens: string[], chunkText: string): number {
  if (queryTokens.length === 0) return 0;
  const chunkLower = chunkText.toLowerCase();
  let matches = 0;
  for (const token of queryTokens) {
    if (chunkLower.includes(token)) {
      matches += 1;
    }
  }
  return matches / queryTokens.length;
}

// Resilient embedding generator with fallback
async function getEmbeddingSafe(ai: GoogleGenAI, text: string): Promise<number[] | null> {
  const embeddingModels = ['gemini-embedding-2-preview', 'text-embedding-004'];
  for (const model of embeddingModels) {
    try {
      const response = await ai.models.embedContent({
        model,
        contents: text,
      });
      const resObj = response as any;
      const values = resObj?.embedding?.values || resObj?.embeddings?.[0]?.values;
      if (Array.isArray(values) && values.length > 0) {
        return values;
      }
    } catch {
      // Continue to next embedding model
    }
  }
  return null;
}

// Core RAG Hybrid Retrieval Function
async function retrieveRelevantChunks(
  ai: GoogleGenAI | null,
  query: string,
  documents: DocumentInput[],
  topK: number = 4
): Promise<KnowledgeChunk[]> {
  if (!documents || documents.length === 0 || !query.trim()) {
    return [];
  }

  // Deconstruct documents into searchable chunks
  const allChunks: KnowledgeChunk[] = [];
  for (const doc of documents) {
    const rawChunks = chunkText(doc.content || '');
    rawChunks.forEach((chunkStr, idx) => {
      allChunks.push({
        chunkId: `${doc.id}_${idx}`,
        docId: doc.id,
        docTitle: doc.title,
        category: doc.category,
        text: chunkStr,
      });
    });
  }

  if (allChunks.length === 0) return [];

  const queryTokens = tokenize(query);

  // Attempt semantic embeddings if GenAI client is available
  let queryEmbedding: number[] | null = null;
  if (ai && query.length < 500) {
    try {
      queryEmbedding = await getEmbeddingSafe(ai, query);
    } catch {
      queryEmbedding = null;
    }
  }

  // Score each chunk
  const scoredChunks: KnowledgeChunk[] = [];
  for (const chunk of allChunks) {
    const lexicalScore = computeLexicalScore(queryTokens, `${chunk.docTitle} ${chunk.text}`);
    let semanticScore = 0;

    if (queryEmbedding && ai) {
      try {
        const chunkEmbedding = await getEmbeddingSafe(ai, chunk.text);
        if (chunkEmbedding) {
          semanticScore = computeCosineSimilarity(queryEmbedding, chunkEmbedding);
        }
      } catch {
        semanticScore = 0;
      }
    }

    // Hybrid combined relevance score (prioritize semantic if available, plus keyword overlap)
    const combinedScore = queryEmbedding
      ? (0.65 * Math.max(0, semanticScore) + 0.35 * lexicalScore)
      : lexicalScore;

    scoredChunks.push({
      ...chunk,
      relevanceScore: Math.round(combinedScore * 100) / 100,
    });
  }

  // Rank descending by score
  scoredChunks.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));

  // Return topK highest scoring chunks
  return scoredChunks.slice(0, topK);
}

// RAG Test / Retrieval Inspection Endpoint
app.post('/api/rag/retrieve', async (req: Request, res: Response): Promise<void> => {
  try {
    const data = (req.body && typeof req.body === 'object') ? req.body : {};
    const query = typeof data.query === 'string' ? data.query : '';
    const documents = Array.isArray(data.documents) ? data.documents : [];
    const topK = typeof data.topK === 'number' ? Math.min(10, Math.max(1, data.topK)) : 4;

    let ai: GoogleGenAI | null = null;
    try {
      ai = getGenAI();
    } catch {
      // Offline / fallback without embedding
    }

    const chunks = await retrieveRelevantChunks(ai, query, documents, topK);
    res.json({
      chunks,
      totalDocuments: documents.length,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('Error in /api/rag/retrieve:', error);
    res.status(500).json({ error: error.message || 'Failed to retrieve relevant chunks' });
  }
});

// Seed Sample Knowledge Endpoint (allows instant 1-click verification of RAG grounding)
app.get('/api/rag/seed-sample', (req: Request, res: Response) => {
  const sampleDocs = [
    {
      id: 'doc_productivity_rules',
      title: 'Daily Deep Work & Productivity Rules',
      category: 'Personal Standards',
      tags: ['productivity', 'habits', 'deep-work', 'standards'],
      content: `Rule 1: Deep Work blocks occur strictly between 08:30 AM and 11:30 AM every weekday. All push notifications, social feeds, and email clients must be silenced.
Rule 2: Daily shutdown ritual begins at 5:30 PM. Any unfinished task must be logged into the backlog, never carried mentally into the evening.
Rule 3: Maximum 3 primary needle-moving priorities per day. Anything beyond 3 is treated as non-essential secondary work.
Rule 4: Physical movement rule: 30 minutes of cardiovascular walking or zone-2 exercise every morning before sitting at the workspace.
Rule 5: No meetings on Wednesdays. Wednesdays are reserved entirely for uninterrupted architecture, research, and coding.`
    },
    {
      id: 'doc_system_architecture',
      title: 'Cloud Run & Firestore Security Architecture Policy',
      category: 'Engineering Policy',
      tags: ['architecture', 'security', 'firestore', 'cloud-run'],
      content: `Standard 1: Zero-Trust Database Isolation. Every user document must be nested strictly under /users/{userId}/. Cross-user access or public read permissions are prohibited.
Standard 2: Payload Sanitization. All objects passed to Firestore setDoc or updateDoc must strictly pass through stripUndefined() to guarantee zero undefined driver crashes.
Standard 3: Gemini Model Ladder. All generative AI calls must route through the resilient fallback ladder: gemini-3.6-flash -> gemini-3.1-flash-lite -> gemini-flash-latest -> gemini-3.7-flash with automatic error recovery.
Standard 4: Secret Management. The GEMINI_API_KEY must never be committed to repository code or exposed to client-side bundles. Secrets must be mounted via Secret Manager or server-side environment variables.`
    },
    {
      id: 'doc_personal_facts',
      title: 'Personal Verified Profile & Core Commitments',
      category: 'Personal Truths',
      tags: ['identity', 'goals', 'profile'],
      content: `Primary Goal for 2026: Launch two independent cloud software platforms that reach $10,000 monthly recurring revenue.
Dietary Constraint: Strict pescatarian; completely allergic to peanuts and shellfish.
Reading Commitment: 24 non-fiction books per year, focused on systems thinking, history, and distributed engineering.
Favorite Philosophy: Marcus Aurelius Stoicism — focus exclusively on what is within direct control, disregard extraneous noise.`
    }
  ];

  res.json({ documents: sampleDocs });
});

// Multi-turn Reflection & Chat Endpoint (with RAG Grounding & Straight-Talk Conditioning)
app.post('/api/gemini/reflect', async (req: Request, res: Response): Promise<void> => {
  try {
    // Defensive payload ingestion
    const data = (req.body && typeof req.body === 'object') ? req.body : {};
    const messages = Array.isArray(data.messages) ? data.messages : [];
    const mode = typeof data.mode === 'string' ? data.mode : 'reflection';
    const journalTopic = typeof data.topic === 'string' ? data.topic : '';
    const ragEnabled = Boolean(data.ragEnabled || mode === 'straight_rag');
    const ragDocuments = Array.isArray(data.ragDocuments) ? data.ragDocuments : [];
    const ragSettings = (data.ragSettings && typeof data.ragSettings === 'object') ? data.ragSettings : {};
    const answeringStyle = typeof ragSettings.answeringStyle === 'string' ? ragSettings.answeringStyle : (mode === 'straight_rag' ? 'straight_talk' : 'standard');
    const antiHallucinationStrictness = ragSettings.antiHallucinationStrictness === 'balanced' ? 'balanced' : 'strict';
    const topK = typeof ragSettings.topK === 'number' ? Math.min(8, Math.max(1, ragSettings.topK)) : 4;

    if (messages.length === 0) {
      res.status(400).json({ error: 'Missing or empty messages array in request body.' });
      return;
    }

    const ai = getGenAI();

    // Identify latest user prompt for retrieval
    const lastUserMessage = [...messages].reverse().find((m: any) => m.role === 'user' || m.role === undefined);
    const queryText = lastUserMessage ? String(lastUserMessage.content || '') : '';

    // Execute RAG retrieval if documents are provided
    let retrievedChunks: KnowledgeChunk[] = [];
    if (ragEnabled && ragDocuments.length > 0 && queryText.trim()) {
      retrievedChunks = await retrieveRelevantChunks(ai, queryText, ragDocuments, topK);
    }

    let systemInstruction = `You are a high-performance AI Journaling, Analysis & Reflection Partner powered by Gemini.`;

    // Anti-Hallucination & Grounding Directives
    if (ragEnabled) {
      systemInstruction += `\n\n### MANDATORY ANTI-HALLUCINATION & FACTUAL GROUNDING DIRECTIVES:
1. You MUST ground your factual assertions strictly and exclusively in the [RETRIEVED VERIFIED KNOWLEDGE BASE CONTEXT] provided below.
2. DO NOT speculate, assume, extrapolate, or hallucinate facts that are not present in the verified context.
3. If the provided context does NOT contain sufficient factual evidence to answer the question, you MUST refuse to fabricate an answer and state directly:
"Based on your verified knowledge base, no factual information is available to answer this question."
4. Whenever you cite or use a fact from the knowledge base, append a clear source citation in brackets, e.g. [Source: Document Title].
5. Never contradict verified facts from the knowledge base.`;

      if (retrievedChunks.length > 0) {
        systemInstruction += `\n\n=== [RETRIEVED VERIFIED KNOWLEDGE BASE CONTEXT] ===\n` +
          retrievedChunks.map((chunk, idx) => 
            `[Source ${idx + 1}: "${chunk.docTitle}" | Category: ${chunk.category || 'General'} | Relevance: ${chunk.relevanceScore || 1.0}]\n${chunk.text}`
          ).join('\n\n---\n\n') +
          `\n=== [END RETRIEVED CONTEXT] ===`;
      } else {
        systemInstruction += `\n\n[NOTICE: The user has enabled RAG grounding, but no matching context chunks were found for this query in their knowledge base. If this is a question about personal facts, rules, or data, explicitly inform them that no matching knowledge base entries were found.]`;
      }
    }

    // Straight-Talk Conditioning Directives
    if (answeringStyle === 'straight_talk' || mode === 'straight_rag') {
      systemInstruction += `\n\n### STRAIGHT TALK & DIRECT ANSWERING DIRECTIVE:
1. State the direct answer clearly in the very FIRST sentence.
2. STERN BAN ON CONVERSATIONAL FILLER: Never start with conversational pleasantries, affirmations, or preamble (e.g. absolutely NO "Sure!", "Certainly!", "I can help with that!", "That is a great question!", "In today's fast-paced world...", or polite throat-clearing).
3. Cut all unnecessary adjectives, repetitive buzzwords, and flattering platitudes.
4. Deliver high-density, actionable, objective information.
5. If bullet points make the information faster to parse, use clean, concise bullet points.`;
    } else if (answeringStyle === 'bulleted') {
      systemInstruction += `\n\nStyle Directive: Structure all findings and takeaways as crisp, high-signal bulleted lists without preamble.`;
    } else if (answeringStyle === 'concise') {
      systemInstruction += `\n\nStyle Directive: Maximum conciseness. Keep answers under 3-4 tight sentences without fluff.`;
    }

    // Secondary Mode Specialization
    if (mode === 'brainstorming') {
      systemInstruction += `\nMode: Brainstorming & Ideas. Focus on creative perspectives, alternative angles, structured action items, and imaginative possibilities.`;
    } else if (mode === 'summary') {
      systemInstruction += `\nMode: Executive Summary & Synthesis. Distill the core themes, key decisions, and takeaways clearly.`;
    } else if (mode === 'action_plan') {
      systemInstruction += `\nMode: Goal & Action Planning. Help convert insights into structured, realistic, step-by-step SMART action items.`;
    } else if (mode === 'emotional_resonance') {
      systemInstruction += `\nMode: Emotional Deep Dive. Validate emotional states, explore subconscious drivers, and foster cognitive reframing.`;
    } else if (mode === 'reflection' && !ragEnabled) {
      systemInstruction += `\nMode: Reflection Companion. Empathetic, supportive, articulate, and objective.`;
    }

    if (journalTopic) {
      systemInstruction += `\nUser's Journal Topic/Theme: "${journalTopic}"`;
    }

    // Format multi-turn conversation for @google/genai
    const formattedContents = messages.map((msg: any) => ({
      role: msg.role === 'assistant' || msg.role === 'model' ? 'model' : 'user',
      parts: [{ text: String(msg.content || '') }]
    }));

    // For strict anti-hallucination & straight talk, lower temperature to prevent random extrapolation
    const temperature = (ragEnabled || mode === 'straight_rag')
      ? (antiHallucinationStrictness === 'strict' ? 0.1 : 0.3)
      : (mode === 'brainstorming' ? 0.8 : 0.6);

    const result = await generateContentWithFallback(ai, formattedContents, {
      systemInstruction,
      temperature,
    });

    const groundingSources = retrievedChunks.map(c => ({
      docId: c.docId,
      docTitle: c.docTitle,
      category: c.category,
      snippet: c.text.length > 220 ? c.text.slice(0, 220) + '...' : c.text,
      relevanceScore: c.relevanceScore,
    }));

    res.json({
      reply: result.text,
      modelUsed: result.modelUsed,
      groundingSources,
      isRagGrounded: retrievedChunks.length > 0,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('Error in /api/gemini/reflect:', error);
    res.status(500).json({
      error: error.message || 'Internal server error while processing reflection with Gemini'
    });
  }
});

// Summarize & Title Generator Endpoint
app.post('/api/gemini/summarize', async (req: Request, res: Response): Promise<void> => {
  try {
    const data = (req.body && typeof req.body === 'object') ? req.body : {};
    const textContent = typeof data.text === 'string' ? data.text : '';
    const messages = Array.isArray(data.messages) ? data.messages : [];

    let combinedText = textContent;
    if (!combinedText && messages.length > 0) {
      combinedText = messages.map((m: any) => `${m.role.toUpperCase()}: ${m.content}`).join('\n\n');
    }

    if (!combinedText.trim()) {
      res.status(400).json({ error: 'Text content or messages required for summarization.' });
      return;
    }

    const ai = getGenAI();

    const prompt = `Analyze the following user journal/reflection session:
---
${combinedText}
---

Provide a JSON response with the following format:
{
  "title": "A concise, evocative 3-6 word title capturing the essence of this reflection",
  "summary": "A clear, 2-3 sentence executive summary of key insights, themes, and realizations",
  "category": "One of: Personal Growth, Career & Work, Relationships, Health & Wellness, Creativity, Gratitude, Decisions",
  "tags": ["3 to 5 relevant short lowercase tags"]
}
Respond with ONLY valid JSON and no code fences or extra text.`;

    const result = await generateContentWithFallback(ai, prompt, {
      temperature: 0.3,
    });

    let cleaned = result.text.trim();
    if (cleaned.startsWith('```')) {
      cleaned = cleaned.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/```\s*$/i, '').trim();
    }

    try {
      const parsed = JSON.parse(cleaned);
      res.json({
        title: parsed.title || 'Journal Reflection',
        summary: parsed.summary || 'A thoughtful personal reflection session.',
        category: parsed.category || 'Personal Growth',
        tags: Array.isArray(parsed.tags) ? parsed.tags : ['reflection', 'journal'],
        modelUsed: result.modelUsed
      });
    } catch {
      res.json({
        title: 'Reflection Session',
        summary: cleaned.slice(0, 200),
        category: 'Personal Growth',
        tags: ['journal', 'reflection'],
        modelUsed: result.modelUsed
      });
    }
  } catch (error: any) {
    console.error('Error in /api/gemini/summarize:', error);
    res.status(500).json({
      error: error.message || 'Failed to generate summary'
    });
  }
});

async function startServer() {
  // Vite middleware setup
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
  });
}

startServer();
