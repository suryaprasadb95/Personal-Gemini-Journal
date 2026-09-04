# Gemini Reflection Journal

A secure, user-authenticated reflection journaling web application powered by **Gemini 3.6 Flash** and **Cloud Firestore** with strict owner-bound data isolation and Google Sign-In authentication.

---

## 1. Environment & Prerequisites

### Required Google Cloud Services & APIs
Ensure the following APIs are enabled in your Google Cloud Project:
- **Cloud Run Admin API** (`run.googleapis.com`)
- **Secret Manager API** (`secretmanager.googleapis.com`)
- **Cloud Firestore API** (`firestore.googleapis.com`)
- **Identity Toolkit API** (`identitytoolkit.googleapis.com` for Firebase Auth)

```bash
# Enable required Google Cloud APIs
gcloud services enable \
  run.googleapis.com \
  secretmanager.googleapis.com \
  firestore.googleapis.com \
  identitytoolkit.googleapis.com
```

---

## 2. Secret Management Setup

Store your Gemini API key securely in Google Cloud Secret Manager instead of hardcoding or exposing it in client code.

```bash
# 1. Create and populate the secret in Secret Manager
gcloud secrets create GEMINI_API_KEY --replication-policy="automatic"
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets versions add GEMINI_API_KEY --data-file=-

# 2. Grant the default Cloud Run service account access to read the secret
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")

gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 3. Database Security Configuration (Cloud Firestore)

Deploy the following security rules to ensure user data isolation so that each authenticated user can only access their own journal entries and interaction history.

### `firestore.rules`
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;

      match /interactions/{interactionId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }

      match /entries/{entryId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }

      match /knowledge_docs/{docId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }

      match /rag_settings/{settingId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }

      match /{allSubcollections=**} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }
    }
  }
}
```

```bash
# Deploy firestore rules using Firebase CLI
firebase deploy --only firestore:rules
```

---

## 4. Local Development & Build

```bash
# Install dependencies
npm install

# Run local development server (Express + Vite on port 3000)
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

---

## 5. Cloud Run Deployment Flow

Deploy the containerized full-stack application directly to Google Cloud Run with the Secret Manager binding and mandatory campaign verification label.

```bash
# Deploy to Google Cloud Run
gcloud run deploy gemini-reflection-journal \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets=GEMINI_API_KEY=GEMINI_API_KEY:latest \
  --update-labels=dev-tutorial=cloud-run-ai-challenge
```

### Required Campaign Labeling
To register the service for automated challenge verification, ensure the label is applied:
```bash
gcloud run services update gemini-reflection-journal \
  --update-labels=dev-tutorial=cloud-run-ai-challenge \
  --region=us-central1
```

---

## 6. RAG Grounding & Anti-Hallucination Engine

The application includes a hybrid Retrieval-Augmented Generation (RAG) system:
- **Hybrid Retrieval**: Combines lexical chunk scoring with semantic embedding similarity to retrieve top-matching knowledge documents.
- **Anti-Hallucination Protocol**: Injects strict grounding directives into Gemini's system instructions with low temperature (`0.1`–`0.3`). If verified facts are absent from the knowledge base, Gemini is instructed to explicitly refuse to speculate.
- **Straight-Talk Conditioning**: Conditions the model to deliver direct, first-sentence answers with zero conversational fluff, disclaimers, or preambles.
- **Source Transparency**: Every grounded response renders an interactive citation card with document titles, relevance scores, and matched text snippets.

---

## 7. Security Architecture & Threat Model

| Threat Zone | Potential Risk | Countermeasure Implemented |
| :--- | :--- | :--- |
| **Input Surfaces** | Malicious injection in journal or knowledge inputs | Strict payload sanitization & defensive null-safe destructuring |
| **Planning & Reasoning** | System prompt escape / Hallucination | Plain-data boundaries, strict RAG grounding directives & anti-hallucination barriers |
| **Tool & Execution** | API key leakage | Gemini API key kept strictly server-side in Express proxy |
| **Memory & State** | Cross-user journal & knowledge leaks | Owner-bound Firestore path security (`request.auth.uid == userId`) for entries & knowledge docs |
| **Inter-System Comms** | Upstream Gemini downtime | Resilient model fallback ladder (`gemini-3.6-flash` &rarr; `gemini-3.1-flash-lite` &rarr; `gemini-flash-latest` &rarr; `gemini-3.7-flash`) |
