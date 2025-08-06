def fuse_predictions(text_emotion, behavior_emotion):
    if text_emotion == behavior_emotion:
        return text_emotion
    return behavior_emotion  # prioritize behavioral truth over words
