# app.py
import streamlit as st
import os
import tempfile
import shutil
import logic

print("Code Running")
st.title("MedSchool IO Card Anki Generator")

# Inputs
api_key = st.text_input("Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload Presentation", type=["pptx"])
start_counter = st.number_input("Last Image # (Start Counter)", value=0, min_value=0)
csv_name = st.text_input("Desired CSV Name", value="deck")

# Execution
if uploaded_file and api_key and st.button("Generate Deck"):

    # Set the API Key for the logic file to use
    groq_api = api_key

    with st.spinner("Processing... This may take a few minutes."):
        # Create a temporary directory (The "Virtual Folder")
        with tempfile.TemporaryDirectory() as temp_dir:

            # Save the uploaded PPTX to this folder
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Run the Pipeline
                logic.run_pipeline(uploaded_file.name, start_counter, temp_dir, groq_api)

                # Run the Formatting
                final_csv = logic.create_csv_file(csv_name, temp_dir)

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


# Instructions
with st.expander("How to import this into Anki?"):
    st.markdown("""
    1. Go to your download folder and unzip your file. The unzipped file should be a folder that contains text files, image files, and a csv. 
    2. If your file does not contain one of those items, refresh the website and recreate your zip file. 
    3. Select all the images in the folder, and press <br>
     Mac: 'command + x' <br>
     Windows/Linux: 'control + x'  
    4. In your files, locate the Anki App
    5. Locate the media folder within your Anki App. It is likely stored in these file locations: <br>
     Windows: '%APPDATA%\Anki2\<Your Profile Name>\collection.media' <br>
     Mac: '~/Library/Application Support/Anki2/<Your Profile Name>/collection.media' <br>
     Linux: '~/.local/share/Anki2/<Your Profile Name>/collection.media' <br>
    If you did not create a profile name, the <Your Profile Name> is likely User 1
    6. Press 
     Mac: 'command + v' <br>
     Windows/Linux: 'control + v'
    7. Now open the Anki
    8. Click import on the top right corner
    9. Find the .csv file in the folder you downloaded, and click on it
    10. Finally, set the type to 'Basic'(The result will be same as if they were Image Occlusion Cards) <br>
    Choose the Deck where you want this to live <br>
    Find the 'Field Separator' and choose Commas <br><br>
    
    ###Congrats you have a new Med School Anki Deck! 
    """)
