Quick Start Guide
=================

This guide will help you get WatchMe AI running in under 10 minutes.

Prerequisites
-------------

Make sure you've completed the :doc:`configuration` steps.

Step 1: Start the Server
------------------------

.. code-block:: bash

   # From the project root
   python main.py

You should see:

.. code-block:: text

   INFO:     Started server process [1234]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:5000

Step 2: Test the API
--------------------

Upload a Reference Image
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   curl -X POST "http://localhost:5000/upload-embedding" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@path/to/person.jpg"

Example response:

.. code-block:: json

   {
     "embedding": "encrypted_embedding_data_here",
     "status": "success"
   }

Search in a Video
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   curl -X POST "http://localhost:5000/find" \
        -H "Content-Type: application/json" \
        -d '{
          "person": {"embed": "encrypted_embedding_data"},
          "video_path": "path/to/video.mp4"
        }'

Example response:

.. code-block:: json

   {
     "matches": [
       {
         "bbox": [100, 150, 200, 400],
         "distance": 0.23
       }
     ]
   }

Step 3: WebSocket Connection
----------------------------

For real-time video processing:

.. code-block:: javascript

   const ws = new WebSocket('ws://localhost:5000/video-stream');

   ws.onopen = function() {
       ws.send(JSON.stringify({
           frame: base64_frame_data,
           person_embed: encrypted_embedding
       }));
   };

   ws.onmessage = function(event) {
       const matches = JSON.parse(event.data);
       console.log('Found matches:', matches);
   };

What's Next?
------------

* :doc:`../guide/training` - Train your own models
* :doc:`../guide/api-endpoints` - Detailed API documentation
* :doc:`../examples/video-search` - More examples
