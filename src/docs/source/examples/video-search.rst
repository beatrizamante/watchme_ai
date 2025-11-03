Video Search Examples
=====================

This guide provides comprehensive examples for searching people in videos using the WatchMe AI Backend, covering both batch processing and real-time scenarios.

Basic Video Search
------------------

Upload Reference Image and Search Video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Upload a reference image to get the person embedding**

.. code-block:: python

   import requests
   import base64

   # Read and encode the reference image
   with open('reference_person.jpg', 'rb') as f:
       image_data = base64.b64encode(f.read()).decode()

   # Upload to get embedding
   response = requests.post('http://localhost:5000/upload-embedding',
                          json={'image': image_data})

   if response.status_code == 200:
       embedding = response.json()['embedding']
       print("✓ Embedding created successfully")
   else:
       print(f"✗ Error: {response.json()['detail']}")

**Step 2: Search for the person in a video**

.. code-block:: python

   # Search in video file
   search_response = requests.post(
       'http://localhost:5000/find',
       json={
           'person': {'embed': embedding},
           'video_path': '/path/to/video.mp4'
       }
   )

   if search_response.status_code == 200:
       results = search_response.json()
       print(f"Found {len(results['matches'])} matches")

       for i, match in enumerate(results['matches']):
           bbox = match['bbox']  # [x1, y1, x2, y2]
           distance = match['distance']
           confidence = match.get('confidence', 0)

           print(f"Match {i+1}:")
           print(f"  Bounding box: {bbox}")
           print(f"  Distance: {distance:.3f}")
           print(f"  Confidence: {confidence:.3f}")
   else:
       print(f"✗ Search failed: {search_response.json()['detail']}")

Complete Python Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import requests
   import base64
   import cv2
   import numpy as np

   class PersonSearcher:
       def __init__(self, base_url="http://localhost:5000"):
           self.base_url = base_url

       def create_embedding(self, image_path):
           """Create person embedding from image"""
           try:
               with open(image_path, 'rb') as f:
                   image_data = base64.b64encode(f.read()).decode()

               response = requests.post(
                   f"{self.base_url}/upload-embedding",
                   json={'image': image_data}
               )

               if response.status_code == 200:
                   return response.json()['embedding']
               else:
                   raise Exception(f"Embedding creation failed: {response.json()['detail']}")

           except Exception as e:
               print(f"Error creating embedding: {e}")
               return None

       def search_in_video(self, embedding, video_path, person_id="default"):
           """Search for person in video"""
           try:
               response = requests.post(
                   f"{self.base_url}/find/{person_id}",
                   json={
                       'person': {'embed': embedding},
                       'video_path': video_path
                   }
               )

               if response.status_code == 200:
                   return response.json()
               else:
                   raise Exception(f"Search failed: {response.json()['detail']}")

           except Exception as e:
               print(f"Error searching video: {e}")
               return None

       def visualize_results(self, video_path, matches):
           """Visualize search results on video"""
           cap = cv2.VideoCapture(video_path)

           if not cap.isOpened():
               print("Error opening video")
               return

           frame_number = 0

           while True:
               ret, frame = cap.read()
               if not ret:
                   break

               # Draw bounding boxes for matches in this frame
               for match in matches:
                   if match.get('frame_number') == frame_number:
                       bbox = match['bbox']
                       distance = match['distance']

                       x1, y1, x2, y2 = map(int, bbox)

                       # Draw rectangle
                       cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                       # Add text
                       text = f"Distance: {distance:.3f}"
                       cv2.putText(frame, text, (x1, y1-10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

               # Display frame
               cv2.imshow('Person Search Results', frame)

               if cv2.waitKey(30) & 0xFF == ord('q'):
                   break

               frame_number += 1

           cap.release()
           cv2.destroyAllWindows()

   # Usage example
   if __name__ == "__main__":
       searcher = PersonSearcher()

       # Create embedding from reference image
       print("Creating person embedding...")
       embedding = searcher.create_embedding("reference_person.jpg")

       if embedding:
           print("✓ Embedding created")

           # Search in video
           print("Searching in video...")
           results = searcher.search_in_video(embedding, "test_video.mp4")

           if results:
               print(f"✓ Found {len(results['matches'])} matches")

               # Visualize results
               searcher.visualize_results("test_video.mp4", results['matches'])
           else:
               print("✗ No matches found")
       else:
           print("✗ Failed to create embedding")
