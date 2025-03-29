import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fer import FER
import tensorflow as tf

# Initialize the FER emotion detector
emotion_detector = FER(mtcnn=True)

def detect_emotion(image):
    try:
        # Detect emotions using FER
        emotions = emotion_detector.detect_emotions(image)
        
        if not emotions:
            return {'neutral': 1.0}, "No faces detected"
        
        # Get emotions for all faces
        all_emotions = []
        for face in emotions:
            emotion_scores = face['emotions']
            # Normalize scores
            max_score = max(emotion_scores.values())
            normalized_scores = {k: v/max_score for k, v in emotion_scores.items()}
            all_emotions.append(normalized_scores)
        
        # Calculate average emotions
        avg_emotions = {}
        for emotion in all_emotions[0].keys():
            avg_emotions[emotion] = sum(face[emotion] for face in all_emotions) / len(all_emotions)
        
        # Determine special mode based on collective emotions
        mode = determine_special_mode(avg_emotions, len(all_emotions))
        
        return avg_emotions, mode
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return {'neutral': 1.0}, "Error in emotion detection"

def determine_special_mode(emotions, num_faces):
    # Define thresholds for different modes
    happy_threshold = 0.7
    sad_threshold = 0.7
    angry_threshold = 0.7
    surprised_threshold = 0.7
    
    if num_faces >= 3:  # Group mode
        if emotions['happy'] > happy_threshold:
            return "🎉 Celebration Mode - Everyone is happy!"
        elif emotions['sad'] > sad_threshold:
            return "😢 Group Sadness Mode - Everyone seems down"
        elif emotions['angry'] > angry_threshold:
            return "😠 Group Tension Mode - High tension in the group"
        elif emotions['surprise'] > surprised_threshold:
            return "😮 Group Surprise Mode - Everyone is surprised!"
    
    return f"👥 Group Analysis Mode - {num_faces} faces detected"

def create_plot(emotion_scores, mode):
    try:
        # Create DataFrame for emotions
        df = pd.DataFrame(list(emotion_scores.items()), columns=['Emotion', 'Confidence'])
        
        # Create plot with improved styling
        plt.figure(figsize=(12, 8))  # Increased height for more space
        
        # Add mode text at the top with more space
        plt.figtext(0.5, 0.95, mode, ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Create bar plot with more space between bars
        bars = plt.bar(df['Emotion'], df['Confidence'], width=0.5)  # Reduced bar width
        
        # Customize the plot
        plt.xlabel("Emotion", fontsize=12, labelpad=15)
        plt.ylabel("Average Confidence", fontsize=12, labelpad=15)
        plt.title("Group Emotion Analysis", fontsize=14, pad=40)  # Increased title padding
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        # Add value labels on top of bars with adjusted position
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,  # Added offset to labels
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=10,
                    rotation=0)
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        
        # Add more padding at the top for the mode text
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95)
        
        # Save the plot with high DPI and proper margins
        plt.savefig("mood_plot.png", dpi=300, bbox_inches='tight', pad_inches=0.8)
        plt.close()
        
        return "mood_plot.png"
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        return None

def analyze_image(image):
    try:
        if image is None:
            return None, None, "No image provided. Please upload an image."
        
        # Detect emotion
        emotion_scores, mode = detect_emotion(image)
        
        # Create and save plot
        plot_path = create_plot(emotion_scores, mode)
        
        # Create emotion report
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        report = f"Mode: {mode}\n\n"
        report += f"Dominant Emotion: {dominant_emotion[0].upper()}\n"
        report += f"Average Confidence: {dominant_emotion[1]:.2f}\n\n"
        report += "Detailed Average Scores:\n"
        for emotion, score in emotion_scores.items():
            report += f"{emotion}: {score:.2f}\n"
        
        return image, plot_path, report
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return None, None, f"Error: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    spacing_size="sm",
    radius_size="md",
    font=["sans-serif", "ui-sans-serif", "system-ui"],
    font_mono=["ui-monospace", "Consolas", "monospace"]
)) as demo:
    gr.Markdown("""
    # 🎭 Group Mood Detection System
    ### Analyze emotions of multiple people in your photos using advanced AI technology
    
    This system can detect various emotions including:
    - 😊 Happiness
    - 😢 Sadness
    - 😠 Anger
    - 😨 Fear
    - 😮 Surprise
    - 😒 Disgust
    - 😐 Neutral
    
    Special Modes:
    - 🎉 Celebration Mode (when everyone is happy)
    - 😢 Group Sadness Mode (when everyone is sad)
    - 😠 Group Tension Mode (when everyone is angry)
    - 😮 Group Surprise Mode (when everyone is surprised)
    
    Upload a photo to get started!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Your Photo")
            upload_input = gr.Image(label="Upload Photo", type="numpy")
            analyze_btn = gr.Button("🔍 Analyze Photo", variant="primary", size="large")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Emotion Analysis Results")
            plot_output = gr.Image(label="Emotion Distribution")
            report_text = gr.Textbox(
                label="Detailed Report",
                lines=5,
                placeholder="Your emotion analysis results will appear here...",
                interactive=False
            )
    
    gr.Markdown("""
    ---
    ### ℹ️ About
    This application uses the FER (Facial Expression Recognition) library to analyze emotions in photos.
    It can detect multiple faces and provide both individual and collective emotion analysis.
    
    *Note: For best results, use clear, well-lit photos with visible faces.*
    """)
    
    # Connect the components
    analyze_btn.click(analyze_image, inputs=[upload_input], outputs=[upload_input, plot_output, report_text])

# Launch the application
if __name__ == "__main__":
    try:
        demo.launch(share=False)
    except Exception as e:
        print(f"Error launching the application: {str(e)}")