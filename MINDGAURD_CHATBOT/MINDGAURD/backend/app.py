from flask import Flask, request, jsonify
from flask_cors import CORS
from model.model_utils import predict_mood_from_text, analyze_mood

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Chat response logic using text model
def get_response(user_input, initial_mood):
    if not user_input:
        return "Please share how you feel!"
    
    # Use text model to predict mood from user input
    user_mood, _ = predict_mood_from_text(user_input)
    if user_mood in ['sad', 'neutral', 'happy']:
        if user_mood == "sad":
            return "I'm sorry to hear that. Maybe take a break, drink some water, or go for a short walk. How do you feel now?"
        elif user_mood == "happy":
            return "Awesome! Keep up the good vibes! Anything else I can help with?"
        else:
            return "Thanks for sharing! How about some deep breathing? Let me know how you feel after!"
    return "Thanks for sharing! Let me know more if you'd like support."

@app.route('/analyze_mood', methods=['POST'])
def analyze_mood_route():
    data = request.get_json()
    mood, message = analyze_mood(data)
    return jsonify({'message': message, 'mood': mood})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    initial_mood = request.args.get('mood', 'neutral')
    response = get_response(user_input, initial_mood)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
