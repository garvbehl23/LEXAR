import streamlit as st

def render_hero_header():
    """Render premium Google DeepMind-style hero header."""
    st.markdown(
        """
        <div style='text-align: center; padding: 4rem 0 3rem 0; position: relative;'>
            <!-- Floating orbs background -->
            <div style='position: absolute; width: 300px; height: 300px; background: radial-gradient(circle, rgba(99, 102, 241, 0.15), transparent); border-radius: 50%; top: -100px; left: 10%; filter: blur(60px); animation: float 8s ease-in-out infinite;'></div>
            <div style='position: absolute; width: 250px; height: 250px; background: radial-gradient(circle, rgba(139, 92, 246, 0.12), transparent); border-radius: 50%; top: -50px; right: 15%; filter: blur(60px); animation: float 10s ease-in-out infinite; animation-delay: -2s;'></div>
            
            <h1 style='
                font-size: 4.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 40%, #a5b4fc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 0.75rem;
                letter-spacing: -0.03em;
                position: relative;
                animation: fadeIn 0.8s ease-out;
            '>
                VakalaatGPT
            </h1>
            
            <p style='
                font-size: 1.25rem;
                color: #9ca3af;
                margin-bottom: 2rem;
                font-weight: 400;
                letter-spacing: 0.01em;
                animation: fadeIn 1s ease-out 0.2s backwards;
            '>
                Interpretable Legal AI for Indian Statutes
            </p>
            
            <div style='
                width: 160px;
                height: 3px;
                background: linear-gradient(90deg, transparent, #6366f1, #8b5cf6, transparent);
                margin: 0 auto 1.5rem auto;
                border-radius: 2px;
                box-shadow: 0 0 30px rgba(99, 102, 241, 0.6);
                animation: fadeIn 1.2s ease-out 0.4s backwards;
            '></div>
            
            <div style='
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(99, 102, 241, 0.08);
                border: 1px solid rgba(99, 102, 241, 0.25);
                padding: 10px 20px;
                border-radius: 24px;
                font-size: 0.9rem;
                color: #c7d2fe;
                font-weight: 500;
                animation: fadeIn 1.4s ease-out 0.6s backwards;
                backdrop-filter: blur(10px);
            '>
                <div style='
                    width: 8px;
                    height: 8px;
                    background: linear-gradient(135deg, #6366f1, #8b5cf6);
                    border-radius: 50%;
                    box-shadow: 0 0 10px rgba(99, 102, 241, 0.8);
                    animation: pulse 2s ease-in-out infinite;
                '></div>
                Powered by LEXAR v1.1
            </div>
        </div>
        
        <style>
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )