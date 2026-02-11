# app.py
import streamlit as st
import os
import tempfile
import shutil
import logic  # Importing your consolidated logic file

st.title("MedSchool IO Card Anki Generator")

# 1. Inputs
api_key = st.text_input("Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload Presentation", type=["pptx"])
start_counter = st.number_input("Last Image # (Start Counter)", value=0, min_value=0)
csv_name = st.text_input("Desired CSV Name", value="deck")

# 2. Execution
if uploaded_file and api_key and st.button("Generate Deck"):

    # Set the API Key for the logic file to use
    os.environ["GROQ_API_KEY"] = api_key

    with st.spinner("Processing... This may take a few minutes."):
        # Create a temporary directory (The "Virtual Folder")
        with tempfile.TemporaryDirectory() as temp_dir:

            # Save the uploaded PPTX to this folder
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # CRITICAL: Switch working directory to the temp folder
            # This makes all the glob.glob("image*.png") calls work locally
            original_dir = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Run the Pipeline (Your Logic)
                logic.run_pipeline(uploaded_file.name, start_counter)

                # Run the Formatting (Your Logic)
                final_csv = logic.create_csv_file(csv_name)

                # Zip Everything for Download
                if final_csv:
                    shutil.make_archive("anki_deck", 'zip', temp_dir)

                    with open("anki_deck.zip", "rb") as f:
                        st.success("Deck Generated Successfully!")
                        st.download_button(
                            label="Download Anki Deck (ZIP)",
                            data=f,
                            file_name=f"{csv_name}_package.zip",
                            mime="application/zip"
                        )
                else:
                    st.error("No cards were generated.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Always return to the original directory
                os.chdir(original_dir)