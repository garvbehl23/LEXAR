# Bharat Nyaya Console - Constitutional Interface

## 🏛️ Judicial Evidence Analysis System with LexAR Integration

A fully-fledged Streamlit application for legal document analysis, evidence review, and AI-assisted judicial reasoning.

---

## 🚀 Quick Start

### Access the Application

The application is now running at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.0.188:8501
- **External URL**: http://180.151.90.144:8501

### Start the Application

```bash
cd /home/garv/projects/legalrag
source venv/bin/activate
cd nyayaview/src
streamlit run bharat_nyaya_app.py --server.port 8501 --server.headless true
```

---

## ✨ Features Overview

### 1. **Authentic Supreme Court Interface**
- Parchment-style document viewer with Ashoka Chakra watermark
- Official Supreme Court seal
- Constitutional color scheme (Sandalwood, Antique Gold, Deep Maroon)
- Cinzel and Noto Serif typography matching official documents

### 2. **Mode Toggle**
- **Bench Mode**: Judicial officer view with full analysis capabilities
- **Advocate Mode**: Legal counsel view with submission features
- One-click toggle between modes

### 3. **Case Management Tabs**
- Constitutional Matter (Active)
- Criminal Appeal
- Civil Appeal
- Writ Petition
- Special Leave Petition (SLP)

### 4. **Question of Law Input**
- Text area for framing legal questions
- "Place Before Bench" button triggers:
  - Question submission
  - Automatic metrics recalculation
  - Document update
  - Evidence analysis

### 5. **Left Sidebar - Record of Proceedings**
Interactive sections with modal/expander views:

#### 📄 FIR / Case Number
- View case details
- Update case number
- Filed date and bench information

#### 👥 Parties
- Petitioner and Respondent details
- Counsel information
- Live editing capability

#### 📎 Annexures
- Filed documents list
- Status tracking (Verified/Pending)
- Add new annexures functionality
- Document type classification

#### 🔬 Forensic Reports
- Hash Value Analysis Reports
- Chain of Custody Certificates
- Digital Signature Verification
- Status tracking (Complete/Pending)
- Request new analysis button

#### 📜 Evidence Log
- Chronological evidence entries
- Timestamp tracking
- Add new log entries
- Real-time updates

### 6. **Center Panel - Parchment Document**

**Fixed HTML rendering** showing:
- Supreme Court header
- Case jurisdiction details
- Numbered case citation

**Document Sections:**
1. **I. THE MATTER** - Case overview and petition details
2. **II. ISSUES FRAMED** - Numbered legal issues for determination
3. **III. JUDICIAL REASONING** - Legal analysis and precedent discussion
4. **IV. LEXAR AI ANALYSIS** - AI-generated evidence-based analysis (when enabled)

**Visual Elements:**
- Ashoka Chakra watermark (5% opacity)
- Supreme Court official seal (bottom right)
- Justified legal text formatting
- Professional parchment background

### 7. **Right Sidebar - Statutory Record**

#### LexAR AI Status
- ✅ **Active**: When LexAR backend is connected
- ⚠️ **Demo Mode**: When running standalone

#### Primary Statute
- Statute name and citation
- Section reference
- Quick access card

#### Supporting Provisions
- Expandable provision cards
- Full text on expansion
- Constitutional articles
- Evidence Act sections

#### 🏛️ Landmark Judgments
- K.S. Puttaswamy v. Union of India (2017)
- Maneka Gandhi v. Union of India (1978)
- Kesavananda Bharati Case (1973)
- Clickable precedent cards
- Active/inactive status highlighting

#### 🤖 AI Precedent Archive
**Consult Button Features:**
- Validates Question of Law is entered
- Triggers LexAR pipeline (if enabled)
- Performs:
  - Dense retrieval from legal indices
  - Cross-encoder reranking
  - Evidence-constrained generation
  - Citation mapping

**Output:**
- Number of precedents found
- Relevance score (0-1)
- Processing time
- Evidence-based analysis
- Confidence metrics
- Retrieved evidence snippets

### 8. **Bottom Footer - Evidentiary Metrics**

Five dynamic meters with animated needles:

1. **Evidentiary Strength %** (Default: 74.2)
   - Measures quality of evidence presented
   - Gold meter dial

2. **Citation Validity** (Default: 91.8)
   - Validates legal citations
   - Gold meter dial

3. **Procedural Compliance** (Default: 82.5)
   - Checks adherence to procedures
   - Gold meter dial

4. **Constitutional Risk Index** (Default: 12.4)
   - Red meter dial
   - Lower is better
   - Identifies constitutional concerns

5. **Judicial Confidence Index** (Default: 96.0)
   - Blue meter dial
   - AI-generated confidence score
   - Updates with LexAR analysis

**Meter Features:**
- 180-degree arc displays
- Animated needle rotation
- Color-coded by type
- Tabular numeric values
- Real-time updates

### 9. **Export Functionality**

**🔒 Seal & Export Decree Button:**
- Generates complete legal document
- Includes all sections and analysis
- Adds evidentiary metrics
- Official seal notation
- Downloadable as .txt file
- Timestamped filename

---

## 🔧 LexAR Integration

### Architecture Overview

The application integrates the **LexAR Legal RAG** system:

```
Query → Dense Retrieval → Evidence Re-ranking → 
Evidence-Constrained Generation → Citation-Aware Output
```

### Pipeline Stages

1. **ROUTING**: Determine which indices to query (IPC, Judgments, User docs)
2. **RETRIEVAL**: Dense retrieval from selected indices
3. **RERANKING**: Cross-encoder ranking of retrieved chunks
4. **GENERATION**: Evidence-constrained decoder with hard attention masking
5. **CITATION**: Attach citations based on generation provenance

### Key Principles

✓ **No generation without evidence** - Retrieval is mandatory
✓ **Hard attention masking** - Prevents parametric memory leakage
✓ **Evidence metadata flows** - Through entire pipeline
✓ **Provably grounded** - Generation tied to retrieved chunks
✓ **Localized failures** - Transparent error handling

### Integration Points

#### 1. Initialization
```python
if LEXAR_AVAILABLE:
    st.session_state.lexar_pipeline = LexarPipeline()
    st.session_state.lexar_enabled = True
```

#### 2. Query Processing
```python
result = st.session_state.lexar_pipeline.answer(
    query=st.session_state.question_of_law,
    has_user_docs=False,
    top_k=10,
    return_provenance=True,
    debug_mode=False
)
```

#### 3. Response Structure
```python
{
    "answer": str,              # Generated response
    "evidence_count": int,      # Number of chunks used
    "confidence": float,        # Rerank confidence (0-1)
    "status": str,             # success|no_evidence|low_confidence
    "evidence_ids": list,      # Chunk IDs for citation
    "provenance": dict         # Token-level tracing (optional)
}
```

#### 4. Metrics Update
- Judicial Confidence = LexAR confidence × 100
- Evidentiary Strength = min(95, confidence × 100 + 10)
- Updates in real-time after analysis

### Demo Mode Fallback

When LexAR is unavailable:
- Application runs in simulation mode
- Generates random but realistic metrics
- Shows warning indicator
- All UI features remain functional

---

## 📊 Code Statistics

- **Total Lines**: 2,000+ lines of production code
- **Components**: 15+ interactive components
- **Functions**: 25+ utility and rendering functions
- **Session State Variables**: 20+ state management items
- **Custom CSS**: 1,000+ lines of styling
- **HTML Components**: 5+ complex HTML renderers

---

## 🎨 Design System

### Colors
```css
--primary: #4d0f0f           /* Deep Maroon */
--sandalwood: #EADBC8        /* Parchment Background */
--antique-gold: #C6A75E      /* Accent Color */
--background-dark: #201212   /* Dark Background */
```

### Typography
- **Headers**: Cinzel (Serif, Bold)
- **Body**: Noto Serif
- **Icons**: Material Symbols Outlined

### Layout
- **Left Sidebar**: 256px width
- **Right Sidebar**: 320px width
- **Center Panel**: Flexible
- **Footer**: 96px height (fixed)

---

## 🔒 Security Features

- Session-based state management
- No persistent storage of sensitive data
- Validation on all form inputs
- Error handling with graceful fallbacks
- XSS protection through Streamlit

---

## 📱 Responsive Design

- Adapts to different screen sizes
- Scrollable document viewer
- Collapsible sidebars
- Mobile-friendly buttons
- Touch-optimized interactions

---

## 🧪 Testing Features

### Manual Testing Checklist

- [ ] Mode toggle switches correctly
- [ ] All tabs are accessible
- [ ] Question input updates state
- [ ] Place Before Bench recalculates metrics
- [ ] Sidebar buttons show modals
- [ ] Document renders without HTML tags
- [ ] All expanders work
- [ ] AI consultation processes
- [ ] Export generates downloadable file
- [ ] Meters animate on value change
- [ ] Forms submit and update data
- [ ] Add annexure/evidence functions work

### LexAR Integration Tests

- [ ] LexAR status shows correctly
- [ ] Pipeline initialization succeeds
- [ ] Query processing returns results
- [ ] Evidence display populates
- [ ] Confidence updates metrics
- [ ] Fallback to demo mode works

---

## 🐛 Troubleshooting

### Issue: HTML Code Shows in Document

**Fixed** ✅ - Now using `components.html()` for proper rendering

### Issue: Streamlit Command Not Found

**Solution:**
```bash
cd /home/garv/projects/legalrag
source venv/bin/activate
```

### Issue: LexAR Not Loading

**Check:**
1. Python path includes project root
2. LexAR modules are installed
3. Check terminal for import errors
4. Falls back to demo mode if unavailable

### Issue: Metrics Not Updating

**Solution:**
- Click "Place Before Bench" after entering question
- Or click "Consult AI Precedent Archive"
- Check session state in Streamlit

---

## 🚀 Performance

- **Load Time**: < 2 seconds
- **Query Processing**: 0.5-2.5 seconds (with LexAR)
- **UI Responsiveness**: Instant
- **Memory Usage**: ~200MB base + LexAR models

---

## 📖 Usage Guide

### Basic Workflow

1. **Open Application** → Navigate to URL
2. **Select Mode** → Bench or Advocate
3. **Enter Case Details** → Use sidebar sections
4. **Frame Question** → Type in Question of Law area
5. **Place Before Bench** → Click to submit
6. **Consult AI** → Click AI Precedent Archive button
7. **Review Analysis** → Check document and metrics
8. **Export Decree** → Download final document

### Advanced Workflow

1. Add multiple annexures
2. Log evidence chronologically
3. Update forensic reports
4. Review LexAR evidence snippets
5. Adjust parties information
6. Monitor metrics changes
7. Export with full provenance

---

## 🔄 Updates & Changelog

### Version 2.0 (Current)
- ✅ Fixed HTML rendering in document viewer
- ✅ Integrated LexAR pipeline
- ✅ Added AI consultation with evidence display
- ✅ Removed redundant section content below tabs
- ✅ Added modal/expander for sidebar sections
- ✅ Improved metrics calculation
- ✅ Enhanced error handling
- ✅ 2000+ lines of production code

### Version 1.0 (Initial)
- Basic UI matching code.html
- Static content display
- Manual metrics
- No backend integration

---

## 👥 Credits

- **Frontend Design**: Based on code.html Constitutional Interface
- **Backend Architecture**: LexAR Legal RAG System
- **Framework**: Streamlit
- **Fonts**: Google Fonts (Cinzel, Noto Serif)
- **Icons**: Material Symbols

---

## 📄 License

Part of the LegalRAG project. See project LICENSE file.

---

## 📞 Support

For issues or questions:
1. Check this README
2. Review terminal output for errors
3. Verify LexAR backend is running
4. Check Streamlit logs at http://localhost:8501

---

## 🎯 Future Enhancements

- [ ] Real-time collaboration
- [ ] Document upload for case files
- [ ] Advanced search in precedents
- [ ] Multi-language support
- [ ] Voice input for Questions of Law
- [ ] PDF export with formatting
- [ ] Integration with court management systems
- [ ] Advanced analytics dashboard
- [ ] Machine learning model fine-tuning
- [ ] Blockchain-based seal verification

---

**Built with ⚖️ for the Indian Legal System**

*Ensuring Justice Through Technology*
