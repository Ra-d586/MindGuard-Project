from utils.predict_text import predict_text_emotion
from utils.predict_behavior import predict_behavior_emotion
from utils.fusion_predict import fuse_predictions
from utils.chatbot import generate_response
from utils.logger import log_feedback

import subprocess

def run_qwen_ollama():
    # Launch Ollama Qwen for free conversation
    print("💬 Switching to Qwen for more open conversation...")
    subprocess.run(["ollama", "run", "qwen"])

def handle_conversation():
    print("Bot: How was your day?")
    user_text = input("You: ")

    # Behavior input
    behavior_data = {
        'HRV': 65,
        'steps': 6000,
        'screen_time': 180,
        'sleep_hours': 6,
        'time_at_home_hrs': 16,
        'time_outside_hrs': 2,
        'calls_count': 5,
        'call_duration': 15,
        'messages_count': 20
    }

    # First prediction
    text_emotion = predict_text_emotion(user_text)
    behavior_emotion = predict_behavior_emotion(behavior_data)

    print(f"🧠 Text Emotion: {text_emotion}")
    print(f"📊 Behavior Emotion: {behavior_emotion}")

    if text_emotion == behavior_emotion:
        print("✅ Emotions match! Proceeding to response and logging...")
        response = generate_response(user_text, text_emotion)
        print("MindPulse:", response)
        log_feedback(behavior_data, text_emotion, user_text)
        run_qwen_ollama()
    else:
        print(f"🤔 You seem to be feeling '{behavior_emotion}' based on your behavior.")
        print("If you're comfortable, tell me more about what happened.")

        user_text_2 = input("You: ")
        second_text_emotion = predict_text_emotion(user_text_2)

        print(f"🔁 Re-checking with new input: '{second_text_emotion}'")

        if second_text_emotion == behavior_emotion:
            print("✅ Emotions now match! Logging and continuing...")
            response = generate_response(user_text_2, behavior_emotion)
            print("MindPulse:", response)
            log_feedback(behavior_data, behavior_emotion, user_text_2)
            run_qwen_ollama()
        else:
            print("😌 It's okay to feel differently.")
            print("If you want to ask anything, go ahead.")
            run_qwen_ollama()

if __name__ == "__main__":
    handle_conversation()
