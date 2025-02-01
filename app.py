import streamlit as st

# Streamlit app title
st.title("Teachable Machine Image Classification in Streamlit")
st.write("Take a picture using your webcam to see predictions from the Teachable Machine model.")

# Use st.camera_input to capture an image
captured_image = st.camera_input("Take a picture")

# If an image is captured, display it and run predictions
if captured_image is not None:
    st.write("### Captured Image")
    st.image(captured_image, caption="Captured Image", use_column_width=True)

    # Convert the captured image to a data URL
    from PIL import Image
    import io
    import base64

    # Open the image and convert it to a base64-encoded string
    img = Image.open(io.BytesIO(captured_image.getvalue()))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{img_str}"

    # Display prediction results
    st.write("### Prediction Results")
    st.write("Running prediction...")

    # Embed TensorFlow.js and Teachable Machine code in an HTML string
    tfjs_code = f"""
    <style>
        #label-container {{
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }}
    </style>
    <div id="label-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>
    <script type="text/javascript">
        // the link to your model provided by Teachable Machine export panel
        const URL = "https://teachablemachine.withgoogle.com/models/EW-D3hPWC/";

        let model, labelContainer, maxPredictions;

        // Load the image model
        async function init() {{
            const modelURL = URL + "model.json";
            const metadataURL = URL + "metadata.json";

            // load the model and metadata
            model = await tmImage.load(modelURL, metadataURL);
            maxPredictions = model.getTotalClasses();

            // Create a label container for displaying predictions
            labelContainer = document.getElementById("label-container");
            for (let i = 0; i < maxPredictions; i++) {{
                labelContainer.appendChild(document.createElement("div"));
            }}

            // Run prediction on the captured image
            await predict();
        }}

        // Run prediction on the captured image
        async function predict() {{
            // Load the captured image
            const img = new Image();
            img.src = "{data_url}";
            await img.decode();

            // Predict the image
            const prediction = await model.predict(img);
            for (let i = 0; i < maxPredictions; i++) {{
                const classPrediction = 
                    prediction[i].className + ": " + (prediction[i].probability * 100).toFixed(2) + "%";
                labelContainer.childNodes[i].innerHTML = classPrediction;
            }}
        }}

        // Initialize the model and run prediction
        init();
    </script>
    """
    # Inject the TensorFlow.js code into the Streamlit app
    st.components.v1.html(tfjs_code, height=500)