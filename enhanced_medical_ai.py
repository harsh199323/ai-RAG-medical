#!/usr/bin/env python3
"""
Enhanced Multi-Agent Medical AI with RAG
24-Hour Vibe Coding Version - Built for medical excellence!
"""

import asyncio
import ollama
import chromadb
from typing import Dict, Any, List, Optional
import json
import time
from pathlib import Path
import sys
import os
from datetime import datetime
import re
import gc
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from contextlib import asynccontextmanager
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("⚠️ FastAPI not available - run: pip install fastapi uvicorn pydantic")

# Streamlit import (optional)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("⚠️ Streamlit not available - run: pip install streamlit")

class EnhancedRAGAgent:
    """Medical agent with RAG capabilities and error handling"""

    def __init__(self, name: str, model: str, specialty: str, chroma_collection):
        self.name = name
        self.model = model
        self.specialty = specialty
        self.collection = chroma_collection  # Shared collection
        self.rag_enabled = chroma_collection is not None
        self.ollama_available = False

        # Initialize Ollama client with error handling
        try:
            self.client = ollama.Client()
            # Test connection with a simple request
            self.client.list()
            self.ollama_available = True
        except Exception as e:
            print(f"⚠️ Ollama connection issue for {name}: {str(e)[:50]}...")
            self.ollama_available = False

    def retrieve_medical_knowledge(self, query: str, n_results: int = 3) -> str:
        """Retrieve relevant medical knowledge from encyclopedias"""
        if not self.rag_enabled:
            return ""

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            if results and results.get("documents") and results["documents"][0]:
                knowledge_context = "\n\n📚 MEDICAL KNOWLEDGE REFERENCES:\n" + "="*50 + "\n"

                for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                    source = meta.get('source', 'Unknown')
                    page = meta.get('page', 'N/A')

                    knowledge_context += f"\n{i}. FROM {source} (Page {page}):\n"
                    knowledge_context += f"   {doc[:300]}...\n"

                return knowledge_context

        except Exception as e:
            print(f"⚠️ RAG retrieval error for {self.name}: {e}")

        return ""

    async def analyze_with_medical_rag(self, patient_info: str) -> Dict[str, Any]:
        """Enhanced medical analysis with comprehensive RAG context"""

        start_time = time.time()

        # Get relevant medical knowledge
        knowledge_context = ""
        if self.rag_enabled:
            # Multiple targeted queries for comprehensive context
            queries = [
                patient_info,  # Direct patient info
                f"{self.specialty.lower()} {patient_info}",  # Specialty-focused
                f"treatment {patient_info}",  # Treatment-focused
            ]

            for query in queries:
                context = self.retrieve_medical_knowledge(query, n_results=2)
                if context:
                    knowledge_context += context + "\n"

        # Build comprehensive medical prompt
        prompt = f"""
You are a {self.specialty} with access to comprehensive medical encyclopedia knowledge.

PATIENT INFORMATION:
{patient_info}

{knowledge_context}

INSTRUCTIONS:
Based on the patient information and the medical knowledge references provided above, please provide your professional analysis focusing on your specialty area.

Structure your response as follows:
1. 🔍 CLINICAL ASSESSMENT: Your primary findings and observations
2. 💡 PROFESSIONAL OPINION: Your expert analysis and reasoning
3. 📋 RECOMMENDATIONS: Specific next steps, treatments, or referrals
4. ⚠️ CONCERNS & RED FLAGS: Any urgent issues or concerns
5. 📚 KNOWLEDGE BASIS: Reference to the medical sources that informed your analysis

IMPORTANT GUIDELINES:
- Be thorough but concise
- Use medical terminology appropriately
- Cite the medical knowledge when applicable
- Consider differential diagnoses where relevant
- Always recommend professional medical consultation for serious concerns
- Maintain professional medical ethics and patient safety standards
"""

        try:
            if not self.ollama_available:
                return {
                    "agent": self.name,
                    "specialty": self.specialty,
                    "model": self.model,
                    "analysis": f"⚠️ Ollama connection not available. Cannot process analysis.\n\nBased on available information:\n{patient_info}\n\nRecommendation: Please ensure Ollama is running and models are loaded.",
                    "response_time": "0.0s",
                    "rag_enabled": self.rag_enabled,
                    "knowledge_retrieved": len(knowledge_context) > 0,
                    "status": "connection_error"
                }

            print(f"🤖 {self.name} analyzing with {'RAG-enhanced' if self.rag_enabled else 'standard'} medical AI...")

            # Try different approaches if one fails
            response = None
            attempts = [
                lambda: self.client.chat(model=self.model, messages=[{"role": "user", "content": prompt}]),
                lambda: self.client.generate(model=self.model, prompt=prompt),
            ]

            for attempt in attempts:
                try:
                    response = attempt()
                    break
                except Exception as e:
                    print(f"   ⚠️ Attempt failed: {str(e)[:50]}...")
                    continue

            if not response:
                raise Exception("All communication attempts with Ollama failed")

            # Extract response content
            if hasattr(response, 'get') and 'message' in response:
                analysis_text = response['message']['content']
            elif hasattr(response, 'get') and 'response' in response:
                analysis_text = response['response']
            else:
                analysis_text = str(response)

            end_time = time.time()

            return {
                "agent": self.name,
                "specialty": self.specialty,
                "model": self.model,
                "analysis": analysis_text,
                "response_time": f"{end_time - start_time:.1f}s",
                "rag_enabled": self.rag_enabled,
                "knowledge_retrieved": len(knowledge_context) > 0,
                "status": "success"
            }

        except Exception as e:
            error_time = time.time() - start_time
            return {
                "agent": self.name,
                "specialty": self.specialty,
                "model": self.model,
                "analysis": f"❌ Analysis failed: {str(e)}\n\n🔄 Try:\n1. Ensure Ollama is running (ollama serve)\n2. Verify model is loaded: ollama list\n3. Test model: ollama run {self.model} 'Hello'\n\nPatient info for manual review:\n{patient_info}",
                "response_time": f"{error_time:.1f}s",
                "rag_enabled": self.rag_enabled,
                "knowledge_retrieved": len(knowledge_context) > 0,
                "status": "error"
            }

class MedicalMultiAgentRAGSystem:
    """Complete medical multi-agent system with RAG enhancement"""

    def __init__(self):
        print("🏥 Initializing Medical AI Agents...")

        # Initialize shared in-memory ChromaDB
        self.chroma_client = None
        self.collection = None
        self.rag_ready = False
        try:
            self.chroma_client = chromadb.Client()  # In-memory for Streamlit compatibility
            self.collection = self.chroma_client.get_or_create_collection(name="medical_knowledge")
            if self.collection.count() == 0:
                print("📚 Ingesting medical knowledge (this may take a few minutes)...")
                self.ingest_medical_knowledge()
            self.rag_ready = self.collection.count() > 0
            print(f"✅ RAG ready with {self.collection.count()} chunks")
        except Exception as e:
            print(f"⚠️ RAG initialization failed: {str(e)[:50]}... Falling back to no RAG.")
            self.rag_ready = False

        # Initialize specialized medical agents with shared collection
        self.agents = [
            EnhancedRAGAgent("Dr. Gemma", "gemma2:9b", "Chief Medical Officer & General Diagnostics", self.collection),
            EnhancedRAGAgent("Dr. Phi", "phi3:3.8b", "Rapid Assessment & Emergency Medicine", self.collection),
            EnhancedRAGAgent("Dr. Llama", "llama3:latest", "Treatment Planning & Patient Care", self.collection)
        ]

        # Check system status
        self.system_ready = any(agent.ollama_available for agent in self.agents)

        print(f"🤖 Agents initialized: {len([a for a in self.agents if a.ollama_available])}/{len(self.agents)} ready")
        print(f"📚 RAG enhancement: {'✅ Enabled' if self.rag_ready else '❌ Disabled'}")

    def ingest_medical_knowledge(self):
        """Adapted from rag_setup_hybrid.py: Optimized ingestion with sampling and deduplication"""
        import PyPDF2

        data_folder = Path("data")
        if not data_folder.exists():
            print("⚠️ Data folder not found - skipping ingestion")
            return

        pdf_files = list(data_folder.glob("*.pdf"))
        if not pdf_files:
            print("⚠️ No PDFs found - skipping ingestion")
            return

        chunk_hashes = set()  # For deduplication
        sampling_rate = 5  # Every 5th page (adjust if boot time is too long)
        max_workers = max(1, os.cpu_count() - 1)
        all_chunks = []

        def clean_text(text: str) -> str:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]]', '', text)
            text = text.replace('', '')
            return text.strip()

        def is_meaningful_chunk(text: str) -> bool:
            if len(text) < 300 or len(text.split()) < 40:
                return False
            skip_phrases = ['table of contents', 'index', 'copyright', 'isbn', 'page number']
            if any(phrase in text.lower() for phrase in skip_phrases):
                return False
            return True

        def semantic_split(text: str) -> List[str]:
            markers = r'(?i)\n\s*(symptoms|treatment|diagnosis|causes|prevention|prognosis|medication|procedure|care plan|assessment|intervention|evaluation|definition|description|purpose|precautions|risks|complications|expected results|resources|key terms):?\s*\n'
            sections = re.split(markers, text)
            return [s.strip() for s in sections if s.strip()]

        def size_split(text: str, max_size: int = 2000) -> List[str]:
            if len(text) <= max_size:
                return [text]
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?=\s[A-Z])|\.\s+', text)
            chunks = []
            current = ""
            for sentence in sentences:
                if len(current + sentence) <= max_size:
                    current += sentence
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sentence
            if current:
                chunks.append(current.strip())
            return chunks

        def deduplicate_chunk(text: str) -> bool:
            chunk_hash = hashlib.md5(text.encode()).hexdigest()
            if chunk_hash in chunk_hashes:
                return False
            chunk_hashes.add(chunk_hash)
            return True

        def process_page(pdf_path: Path, page_num: int) -> List[Dict[str, Any]]:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = reader.pages[page_num].extract_text()
                    if not text or len(text.strip()) < 100:
                        return []

                cleaned = clean_text(text)
                if not cleaned:
                    return []

                semantic_chunks = semantic_split(cleaned)
                final_chunks = []

                for chunk in semantic_chunks:
                    size_chunks = size_split(chunk)
                    for sc in size_chunks:
                        if is_meaningful_chunk(sc) and deduplicate_chunk(sc):
                            final_chunks.append(sc)

                chunk_dicts = []
                for i, chunk_text in enumerate(final_chunks):
                    chunk_id = f"{pdf_path.stem}_p{page_num:04d}_c{i:03d}"
                    chunk_dicts.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": {"source": pdf_path.name, "page": page_num + 1, "length": len(chunk_text)}
                    })

                return chunk_dicts

            except Exception:
                return []

        start_time = time.time()
        for pdf in pdf_files:
            print(f"📖 Sampling {pdf.name}...")
            try:
                reader = PyPDF2.PdfReader(pdf)
                total_pages = len(reader.pages)
                sampled_pages = list(range(0, total_pages, sampling_rate))
                print(f"   📄 Sampling {len(sampled_pages)}/{total_pages} pages")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_page, pdf, i) for i in sampled_pages]
                    for future in as_completed(futures):
                        all_chunks.extend(future.result())
                        gc.collect()

            except Exception as e:
                print(f"⚠️ Error ingesting {pdf.name}: {str(e)[:50]}...")

        if all_chunks:
            documents = [c["text"] for c in all_chunks]
            ids = [c["id"] for c in all_chunks]
            metadatas = [c["metadata"] for c in all_chunks]

            batch_size = 800  # Smaller batches for memory safety
            for i in range(0, len(documents), batch_size):
                try:
                    self.collection.add(
                        documents=documents[i:i+batch_size],
                        ids=ids[i:i+batch_size],
                        metadatas=metadatas[i:i+batch_size]
                    )
                    gc.collect()
                except Exception as e:
                    print(f"⚠️ Batch add failed: {str(e)[:50]}...")

        total_time = time.time() - start_time
        print(f"✅ Ingested {self.collection.count()} unique chunks in {total_time/60:.1f} min")

    async def comprehensive_medical_analysis(self, patient_info: str) -> Dict[str, Any]:
        """Run comprehensive medical analysis with all available agents"""
        print("🏥 Starting Comprehensive Medical Analysis...")

        if not self.system_ready:
            return {
                "patient_info": patient_info,
                "agent_results": [],
                "summary": "❌ System not ready. Please ensure Ollama is running and models are available.",
                "rag_status": self.rag_ready,
                "system_status": "not_ready"
            }

        # Run all available agents concurrently
        available_agents = [agent for agent in self.agents if agent.ollama_available]
        tasks = [agent.analyze_with_medical_rag(patient_info) for agent in available_agents]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and process results
            valid_results = []
            for result in results:
                if isinstance(result, dict):
                    valid_results.append(result)
                else:
                    print(f"⚠️ Agent error: {result}")

            # Generate integrated medical summary
            summary = await self.generate_medical_summary(valid_results, patient_info)

            return {
                "patient_info": patient_info,
                "agent_results": valid_results,
                "summary": summary,
                "rag_status": self.rag_ready,
                "system_status": "operational",
                "agents_used": len(valid_results)
            }

        except Exception as e:
            return {
                "patient_info": patient_info,
                "agent_results": [],
                "summary": f"❌ Analysis failed: {str(e)}",
                "rag_status": self.rag_ready,
                "system_status": "error"
            }

    async def generate_medical_summary(self, agent_results: list, patient_info: str) -> str:
        """Generate integrated medical summary from all agent analyses"""
        if not agent_results:
            return "❌ No agent analyses available for summary generation."

        try:
            # Use the most capable available agent for summary
            summary_agent = None
            for agent in self.agents:
                if agent.ollama_available and "gemma2" in agent.model.lower():
                    summary_agent = agent
                    break

            if not summary_agent:
                summary_agent = next((a for a in self.agents if a.ollama_available), None)

            if not summary_agent:
                return "❌ No agents available for summary generation."

            # Get additional context for comprehensive summary
            summary_context = ""
            if summary_agent.rag_enabled:
                summary_context = summary_agent.retrieve_medical_knowledge(
                    f"medical summary differential diagnosis {patient_info}", n_results=2
                )

            # Combine all analyses
            combined_analyses = "\n\n" + "="*60 + "\n"
            for i, result in enumerate(agent_results, 1):
                combined_analyses += f"\n{i}. ANALYSIS FROM {result['agent']} ({result['specialty']}):\n"
                combined_analyses += f"Status: {result.get('status', 'unknown')}\n"
                combined_analyses += f"Response Time: {result['response_time']}\n"
                combined_analyses += f"RAG Enhanced: {'Yes' if result['knowledge_retrieved'] else 'No'}\n"
                combined_analyses += f"\nAnalysis:\n{result['analysis']}\n"
                combined_analyses += "="*60 + "\n"

            summary_prompt = f"""
As Chief Medical Officer, provide a comprehensive integrated medical summary based on these specialist consultations:

PATIENT CASE:
{patient_info}

SPECIALIST CONSULTATIONS:
{combined_analyses}

{summary_context}

GENERATE INTEGRATED MEDICAL SUMMARY:

🏥 **INTEGRATED MEDICAL ASSESSMENT**

1. **CONSENSUS FINDINGS**: What do all specialists agree on?
2. **KEY CLINICAL INSIGHTS**: Most important medical observations
3. **PRIORITY RECOMMENDATIONS**: Immediate and follow-up actions
4. **RISK ASSESSMENT**: Critical concerns and safety considerations
5. **DIFFERENTIAL CONSIDERATIONS**: Alternative diagnoses to consider
6. **NEXT STEPS**: Clear action plan for patient care
7. **SPECIALIST COORDINATION**: Any conflicting opinions requiring resolution

Maintain medical professionalism while being accessible to healthcare teams.
Include confidence levels and emphasize areas requiring immediate attention.
"""

            response = summary_agent.client.chat(
                model=summary_agent.model,
                messages=[{"role": "user", "content": summary_prompt}]
            )

            if hasattr(response, 'get') and 'message' in response:
                return response['message']['content']
            else:
                return str(response)

        except Exception as e:
            return f"❌ Summary failed: {str(e)}"

# ===== FASTAPI MODELS AND ENDPOINTS =====

# Pydantic models for API
class PatientCaseRequest(BaseModel):
    patient_info: str

class AnalysisResponse(BaseModel):
    patient_info: str
    agent_results: List[Dict[str, Any]]
    summary: str
    rag_status: bool
    system_status: str
    agents_used: int
    timestamp: str

class SystemStatus(BaseModel):
    system_ready: bool
    rag_ready: bool
    agents_available: int
    models_status: Dict[str, bool]

# Global system instance
medical_system: Optional[MedicalMultiAgentRAGSystem] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system on startup"""
    global medical_system
    medical_system = MedicalMultiAgentRAGSystem()
    yield
    # Cleanup on shutdown
    medical_system = None

def create_fastapi_app() -> FastAPI:
    """Create FastAPI application with medical analysis endpoints"""

    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not available. Install with: pip install fastapi uvicorn pydantic")

    app = FastAPI(
        title="Medical AI RAG System API",
        description="Multi-Agent Medical AI System with RAG-enhanced analysis",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS middleware for React frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"],  # React dev server
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", summary="Health check endpoint")
    async def health_check():
        """Check system status and availability"""
        if not medical_system:
            return {"status": "initializing"}

        return {
            "status": "healthy" if medical_system.system_ready else "degraded",
            "system_ready": medical_system.system_ready,
            "rag_ready": medical_system.rag_ready,
            "agents_available": len([a for a in medical_system.agents if a.ollama_available]),
            "timestamp": datetime.now().isoformat()
        }

    @app.get("/status", response_model=SystemStatus, summary="Detailed system status")
    async def get_system_status():
        """Get detailed system status including model availability"""
        if not medical_system:
            raise HTTPException(status_code=503, detail="System not initialized")

        models_status = {}
        try:
            client = ollama.Client()
            model_list = client.list()
            available_models = [m.get('name', '') for m in model_list.get('models', [])]

            for agent in medical_system.agents:
                models_status[agent.model] = agent.model in available_models
        except:
            # If Ollama is not available, mark all as false
            for agent in medical_system.agents:
                models_status[agent.model] = False

        return SystemStatus(
            system_ready=medical_system.system_ready,
            rag_ready=medical_system.rag_ready,
            agents_available=len([a for a in medical_system.agents if a.ollama_available]),
            models_status=models_status
        )

    @app.post("/analyze", response_model=AnalysisResponse, summary="Analyze patient case")
    async def analyze_patient_case(request: PatientCaseRequest):
        """Perform comprehensive multi-agent medical analysis"""
        if not medical_system:
            raise HTTPException(status_code=503, detail="System not initialized")

        if not medical_system.system_ready:
            raise HTTPException(
                status_code=503,
                detail="Medical AI system not ready. Ensure Ollama is running and models are loaded."
            )

        try:
            result = await medical_system.comprehensive_medical_analysis(request.patient_info)

            return AnalysisResponse(
                patient_info=result["patient_info"],
                agent_results=result["agent_results"],
                summary=result["summary"],
                rag_status=result["rag_status"],
                system_status=result["system_status"],
                agents_used=result["agents_used"],
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    @app.get("/models", summary="List available Ollama models")
    async def list_models():
        """List available Ollama models"""
        try:
            client = ollama.Client()
            model_list = client.list()
            return {
                "models": [m.get('name', '') for m in model_list.get('models', [])],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": f"Failed to connect to Ollama: {str(e)}",
                "models": [],
                "timestamp": datetime.now().isoformat()
            }

    return app

# Streamlit Medical Interface
def create_medical_interface():
    """Professional medical AI interface"""

    if not HAS_STREAMLIT:
        print("❌ Streamlit not available. Install with: pip install streamlit")
        return

    # Page configuration
    st.set_page_config(
        page_title="Medical AI RAG System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for medical theme
    st.markdown("""
    <style>
    .main-header {
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #e1f5fe;
    }
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .status-good { color: #4caf50; }
    .status-warning { color: #ff9800; }
    .status-error { color: #f44336; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header"><h1>🏥 Medical AI RAG System</h1><p>Enhanced Multi-Agent Medical Analysis with Encyclopedia Knowledge</p></div>', unsafe_allow_html=True)

    # Sidebar - System Status
    with st.sidebar:
        st.header("🔧 System Status")

        # Check Ollama
        st.subheader("🤖 AI Models")
        models_to_check = ["gemma2:9b", "phi3:3.8b", "llama3:latest"]
        ollama_working = False

        try:
            client = ollama.Client()
            available_models = []
            try:
                model_list = client.list()
                available_models = [m.get('name', '') for m in model_list.get('models', [])]
                ollama_working = len(available_models) > 0
            except:
                pass

            for model in models_to_check:
                if any(model in available for available in available_models):
                    st.markdown(f'<span class="status-good">✅ {model}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="status-error">❌ {model}</span>', unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<span class="status-error">❌ Ollama: Connection failed</span>', unsafe_allow_html=True)

        # Check RAG Database
        st.subheader("📚 Medical Knowledge")
        try:
            from chromadb import Client as ChromaClient
            chroma_client = ChromaClient()
            collection = chroma_client.get_or_create_collection(name="medical_knowledge")
            count = collection.count()
            st.markdown(f'<span class="status-good">✅ RAG Database (In-Memory): {count:,} chunks</span>', unsafe_allow_html=True)
            rag_available = True
        except Exception as e:
            st.markdown(f'<span class="status-warning">⚠️ RAG: Not available</span>', unsafe_allow_html=True)
            st.info("RAG will initialize on first analysis")
            rag_available = False

        # System recommendations
        st.subheader("💡 Quick Actions")
        if not ollama_working:
            st.error("Start Ollama: `ollama serve`")
        if not rag_available:
            st.info("Knowledge base loads on-demand in RAM")

        if ollama_working:
            st.success("🎉 System ready!")
            if st.button("🧪 Test Analysis"):
                st.session_state.test_analysis = True

    # Main interface
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📝 Patient Information")

        # Medical case templates
        case_templates = {
            "Select a medical case...": "",
            "🫀 Cardiac - Chest Pain": "67-year-old male presents with acute onset chest pain radiating to left arm, accompanied by shortness of breath, diaphoresis, and nausea. Patient has history of hypertension, diabetes, and smoking. Pain started 2 hours ago while watching television. Vital signs: BP 160/95, HR 102, RR 20, O2 sat 94% on room air.",

            "🧠 Neurological - Severe Headache": "34-year-old female presents with sudden onset severe headache described as 'worst headache of my life', accompanied by photophobia, nausea, vomiting, and neck stiffness. No history of migraines. Onset 6 hours ago. Patient appears distressed and prefers dark room. No focal neurological deficits noted.",

            "🦴 Rheumatological - Joint Pain": "45-year-old female presents with bilateral knee and wrist pain, morning stiffness lasting 3+ hours, fatigue, and low-grade fever for 6 weeks. Joint swelling and warmth noted. Family history of autoimmune disease. ESR and CRP elevated. Patient reports difficulty with daily activities.",

            "🫁 Respiratory - Shortness of Breath": "58-year-old male with 30-pack-year smoking history presents with progressive shortness of breath, chronic productive cough with blood-tinged sputum, unintentional 15-pound weight loss over 3 months. Chest X-ray shows lung mass. Patient appears cachectic and anxious.",

            "🩸 Endocrine - Diabetes Symptoms": "42-year-old obese female presents with polyuria, polydipsia, polyphagia, fatigue, and blurred vision for 2 months. Random glucose 285 mg/dL, HbA1c 11.2%. Family history significant for Type 2 diabetes. BMI 32. No ketones in urine.",

            "Custom Case": "custom"
        }

        selected_template = st.selectbox("📋 Medical Case Templates:", list(case_templates.keys()))

        if selected_template == "Custom Case":
            patient_info = st.text_area(
                "Enter patient information:",
                height=200,
                placeholder="Describe patient presentation, symptoms, history, physical findings, and any relevant clinical data..."
            )
        elif case_templates[selected_template]:
            patient_info = st.text_area(
                "Patient Information:",
                value=case_templates[selected_template],
                height=200
            )
        else:
            patient_info = st.text_area(
                "Patient Information:",
                height=200,
                placeholder="Select a template or enter custom patient information..."
            )

        # Analysis button
        analyze_button = st.button("🔬 Analyze Patient Case", type="primary", disabled=not patient_info.strip())

        if analyze_button and patient_info.strip():
            # Store analysis in session state to persist during reruns
            with st.spinner("🤖 Medical AI agents are analyzing the case..."):
                system = MedicalMultiAgentRAGSystem()
                st.session_state.analysis_report = asyncio.run(
                    system.comprehensive_medical_analysis(patient_info)
                )

    with col2:
        st.header("📊 Medical Analysis Results")

        # Display test analysis if triggered
        if hasattr(st.session_state, 'test_analysis') and st.session_state.test_analysis:
            with st.spinner("🧪 Running test analysis..."):
                test_system = MedicalMultiAgentRAGSystem()
                test_patient = "Patient presents with fever, cough, and fatigue. Possible viral infection."
                test_result = asyncio.run(test_system.comprehensive_medical_analysis(test_patient))

                st.subheader("🧪 Test Analysis Results")
                if test_result["system_status"] == "operational":
                    st.success("✅ System is working! RAG initialized successfully.")
                    st.metric("RAG Chunks Loaded", test_system.collection.count() if test_system.collection else 0)
                else:
                    st.error("❌ Test failed - check system setup")

                st.session_state.test_analysis = False
                st.rerun()

        # Display analysis if available
        if hasattr(st.session_state, 'analysis_report'):
            report = st.session_state.analysis_report

            # System status indicator
            if report["system_status"] == "operational":
                st.success(f"✅ Analysis Complete - {report['agents_used']} AI specialists consulted")
            else:
                st.error("❌ System Error - Check Ollama connection")
                st.info("💡 Ensure Ollama is running: `ollama serve`")

            # RAG enhancement status
            if report["rag_status"]:
                st.info("📚 Analysis enhanced with medical encyclopedia knowledge (in-memory)")
            else:
                st.warning("⚠️ Running without RAG enhancement")

            # Individual agent results
            if report["agent_results"]:
                st.subheader("👨‍⚕️ Specialist Consultations")

                for result in report["agent_results"]:
                    with st.expander(f"🤖 {result['agent']} - {result['specialty']}", expanded=True):
                        # Agent metadata
                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("Model", result['model'])
                        col_b.metric("Response Time", result['response_time'])
                        col_c.metric("RAG Enhanced", "✅" if result['knowledge_retrieved'] else "❌")
                        col_d.metric("Status", "✅" if result.get('status') == 'success' else "⚠️")

                        # Analysis content
                        st.markdown("**Medical Analysis:**")
                        st.markdown(result["analysis"])

                # Integrated summary
                st.subheader("🏥 Integrated Medical Summary")
                st.markdown(report["summary"])

                # Export functionality
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    if st.button("💾 Export Full Report"):
                        report_json = json.dumps(report, indent=2, default=str)
                        st.download_button(
                            label="📥 Download JSON Report",
                            data=report_json,
                            file_name=f"medical_analysis_{int(time.time())}.json",
                            mime="application/json"
                        )

                with col_export2:
                    if st.button("📋 Copy Summary"):
                        st.code(report["summary"])

            else:
                st.warning("⚠️ No analysis results available")

        else:
            # Default state
            st.info("👆 Select a medical case and click 'Analyze Patient Case' to begin")
            st.markdown("""
            **🏥 Medical AI RAG System Features:**
            - 🤖 Multiple AI medical specialists
            - 📚 Enhanced with medical encyclopedia knowledge (loaded in RAM)
            - 🔬 Comprehensive clinical analysis
            - 📊 Integrated medical summaries
            - 💾 Exportable reports
            - 🚨 Safety-focused recommendations
            - ⚡ Optimized for Streamlit Community Cloud (no disk limits)
            """)

# Command line interface for testing
def cli_interface():
    """Command line interface for quick testing"""
    print("🏥 Medical AI RAG System - CLI Mode")
    print("=" * 50)

    system = MedicalMultiAgentRAGSystem()

    if not system.system_ready:
        print("❌ System not ready. Please check Ollama connection.")
        return

    print("📝 Enter patient information (or 'quit' to exit):")

    while True:
        patient_info = input("\nPatient case: ").strip()

        if patient_info.lower() in ['quit', 'exit', 'q']:
            print("👋 Medical AI RAG system shutting down.")
            break

        if not patient_info:
            continue

        print("\n🤖 Analyzing...")
        try:
            report = asyncio.run(system.comprehensive_medical_analysis(patient_info))

            print(f"\n📊 MEDICAL ANALYSIS REPORT")
            print("=" * 60)
            print(f"Agents consulted: {report.get('agents_used', 0)}")
            print(f"RAG enhanced: {'Yes' if report['rag_status'] else 'No'}")

            for result in report.get("agent_results", []):
                print(f"\n👨‍⚕️ {result['agent']} ({result['response_time']}):")
                print("-" * 40)
                print(result["analysis"][:500] + "..." if len(result["analysis"]) > 500 else result["analysis"])

            print(f"\n🏥 INTEGRATED SUMMARY:")
            print("=" * 40)
            print(report["summary"])

        except Exception as e:
            print(f"❌ Analysis failed: {e}")

def run_fastapi_server():
    """Run FastAPI server with uvicorn"""
    try:
        import uvicorn
        app = create_fastapi_app()
        print("🏥 Starting Medical AI RAG System API Server...")
        print("📡 API available at: http://localhost:8000")
        print("📚 API documentation at: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("❌ uvicorn not available. Install with: pip install uvicorn")
    except Exception as e:
        print(f"❌ Failed to start FastAPI server: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--cli":
            cli_interface()
        elif sys.argv[1] == "--api":
            run_fastapi_server()
        elif sys.argv[1] == "--streamlit" and HAS_STREAMLIT:
            create_medical_interface()
        else:
            print("🏥 Medical AI RAG System")
            print("Usage:")
            print("  python enhanced_medical_ai.py --api        # Run FastAPI server (for React frontend)")
            print("  python enhanced_medical_ai.py --cli        # Run CLI interface")
            print("  python enhanced_medical_ai.py --streamlit  # Run Streamlit interface")
            print("  python enhanced_medical_ai.py              # Auto-detect mode")
    elif HAS_STREAMLIT:
        # Default to Streamlit if available
        print("🎨 Starting Streamlit interface...")
        create_medical_interface()
    elif HAS_FASTAPI:
        # Fall back to FastAPI server
        print("🚀 Starting FastAPI server...")
        run_fastapi_server()
    else:
        print("🏥 Medical AI RAG System")
        print("Options:")
        print("1. Install FastAPI: pip install fastapi uvicorn pydantic")
        print("2. Install Streamlit: pip install streamlit")
        print("3. Run API: python enhanced_medical_ai.py --api")
        print("4. Run CLI: python enhanced_medical_ai.py --cli")

        # If no CLI args and no UI frameworks, run CLI
        cli_interface()
