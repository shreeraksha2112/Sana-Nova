from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# LLM Configuration
llm_diagnosis = ChatGroq(
    api_key="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",
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

# Prompt Template
prompt_template_diagnosis = PromptTemplate(
    input_variables=['symptoms'],
    template=(
        "Advanced Medical Diagnostic Assistant:\n\n"
        "Symptoms Provided: {symptoms}\n\n"
        "Based on the symptoms, generate a detailed diagnostic report in the following format:\n\n"
        "Diseases:\n"
        "- Disease Name 1\n"
        "  - Common Drug: drug_name_1\n"
        "  - Recommended Tests:\n"
        "    - Test 1: expected finding\n"
        "    - Test 2: expected finding\n"
        "  - Reason: Explain why this disease is suspected.\n\n"
        "- Disease Name 2\n"
        "  - Common Drug: drug_name_2\n"
        "  - Recommended Tests:\n"
        "    - Test 1: expected finding\n"
        "    - Test 2: expected finding\n"
        "  - Reason: Explain how the symptoms relate to this condition.\n\n"
        "- Rare Disease Name 3 (Rare)\n"
        "  - Common Drug: drug_name_3\n"
        "  - Recommended Tests:\n"
        "    - Test 1: expected finding\n"
        "    - Test 2: expected finding\n"
        "  - Reason: Explain why this rare disease is still a possible match based on symptoms.\n\n"
        "Ensure your recommendations are medically plausible and helpful for further diagnosis."
    )
)

# Chain setup
chain = LLMChain(llm=llm_diagnosis, prompt=prompt_template_diagnosis)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '').strip()
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # Run LLM
        input_data = {'symptoms': symptoms}
        results = chain.run(input_data)
        
        # Extract using regex
        disease_blocks = re.findall(
            r'- (.*?)\n\s+- Common Drug: (.*?)\n\s+- Recommended Tests:\n(.*?)\n\s+- Reason: (.*?)\n',
            results,
            re.DOTALL
        )
        
        # Structure the response
        diagnosis_report = []
        for disease, drug, tests_block, reason in disease_blocks:
            tests = [test.strip('- ').strip() for test in tests_block.strip().split('\n')]
            diagnosis_report.append({
                'disease': disease.strip(),
                'drug': drug.strip(),
                'tests': tests,
                'reason': reason.strip()
            })
        
        return jsonify({
            'success': True,
            'report': diagnosis_report
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)