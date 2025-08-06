def generate_response(text, emotion):
    templates = {
        "happy": "That's great to hear! Keep up the good vibes.",
        "sad": "I'm here for you. Want to talk about what’s making you feel down?",
        "anxious": "It’s okay to feel anxious. Would you like a tip to calm down?",
        "angry": "It's okay to be upset. Take a moment, maybe breathe a little.",
        "neutral": "I'm listening. How can I support you today?"
    }
    return templates.get(emotion, "I'm here to support you. Want to talk more?")
