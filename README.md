# Emotion_detection-using-RESnet

Starting with a base model: Instead of building model from scratch, which takes a lot of time and data, we took a shortcut. We used a very powerful setup called ResNet50. This setup was already trained on millions of different images to recognize all sorts of everyday objects. Think of it like giving our program a strong head start in "seeing" things in general.

Teaching It Emotions: ResNet50 is great at recognizing general objects, but it doesn't know anything about human emotions. So, we added our own special parts to ResNet50. These parts are like extra "filters" that learn to focus specifically on facial expressions and guess the emotion.

Fine-Tuning (Optional): Sometimes, after adding our new parts, we let the main ResNet50 section learn a little bit more about emotions, but very carefully. This helps it work even better for our specific task without forgetting its original strong skills.

The Result: The finished program takes your image, runs it through the smart ResNet50 part, then through our custom emotion-guessing filters. What comes out is its best guess as a number, telling us which emotion it thinks is in the picture.
