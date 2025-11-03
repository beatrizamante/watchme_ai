API Endpoints Reference
=======================

This document provides a comprehensive reference for all WatchMe AI Backend API endpoints, including request/response formats, authentication, and usage examples.

Base URL
--------

.. code-block:: text

   http://localhost:5000

All endpoints are prefixed with the base URL. For production deployments, replace with your actual domain.

Authentication
--------------

Currently, the AI Backend does not require authentication, but it's designed to integrate with the WatchMe TypeScript API which handles user authentication via JWT tokens.

Content Types
-------------

- **Request:** ``application/json`` or ``multipart/form-data``
- **Response:** ``application/json``
- **WebSocket:** ``application/json`` messages

Core Endpoints
--------------

Upload Person Embedding
~~~~~~~~~~~~~~~~~~~~~~~

Create an encrypted embedding from a person image.

**Endpoint:** ``POST /upload-embedding``

**Request Body:**

.. code-block:: json

   {
     "image": "base64_encoded_image_string"
   }

**Response:**

.. code-block:: json

   {
     "embedding": "base64_encoded_encrypted_embedding",
     "shape": [512],
     "dtype": "float32",
     "status": "success"
   }

**Example:**

.. code-block:: bash

   curl -X POST "http://localhost:5000/upload-embedding" \
        -H "Content-Type: application/json" \
        -d '{
          "image": "/9j/4AAQSkZJRgABAQEAYABgAAD..."
        }'

**Error Responses:**

.. code-block:: json

   {
     "detail": "No person detected, please try with another image",
     "status": "error"
   }

Find Person in Video
~~~~~~~~~~~~~~~~~~~~

Search for a person in a video using their embedding.

**Endpoint:** ``POST /find``

**Request Body:**

.. code-block:: json

   {
      "person": {
        "id": number;
        "name": string;
        "user_id": number;
        "embedding": Blob
      },
      "video": {
        "id": number;
        "user_id": number;
        "path": string;
      }
   }

**Response:**

.. code-block:: json

   {
     "matches": [
       {
         "bbox": [100, 150, 200, 400],
         "distance": 0.23,
         "confidence": 0.85,
         "frame_number": 45
       }
     ],
     "total_matches": 1,
     "processing_time": 2.34
   }

**Example:**

.. code-block:: bash

   curl -X POST "http://localhost:5000/find" \
        -H "Content-Type: application/json" \
        -d '{
          "person": {
            ...person,
            "embedding": "gAAAAABhN2K8..."
          },
          "video": {
            ...video,
            "path": "uploads/video.mp4"
          }
        }'

WebSocket Endpoints
-------------------

Real-time Video Stream Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Endpoint:** ``ws://localhost:5000/video-stream/{session_id}``

**Connection Parameters:**

- ``session_id`` (string): Unique identifier for the session

**Message Types:**

**Start Tracking:**

.. code-block:: json

   {
     "type": "start_tracking",
     "person_embed": "base64_encoded_encrypted_embedding",
     "video_source": 0,
     "fps_limit": 10
   }

**Single Frame Processing:**

.. code-block:: json

   {
     "type": "single_frame",
     "frame": "base64_encoded_frame",
     "person_embed": "base64_encoded_encrypted_embedding"
   }

**Stop Tracking:**

.. code-block:: json

   {
     "type": "stop_tracking"
   }

**Response Messages:**

**Matches Found:**

.. code-block:: json

   {
     "type": "matches",
     "matches": [
       {
         "bbox": [100, 150, 200, 400],
         "distance": 0.23
       }
     ],
     "timestamp": 1635789012.34,
     "frame_info": {
       "width": 1920,
       "height": 1080,
       "frame_number": 120
     }
   }

**Status Updates:**

.. code-block:: json

   {
     "type": "status",
     "status": "tracking_started",
     "message": "Started tracking on source: 0",
     "timestamp": 1635789012.34
   }

**JavaScript Example:**

.. code-block:: javascript

   const ws = new WebSocket('ws://localhost:5000/video-stream/session123');

   ws.onopen = function() {
       ws.send(JSON.stringify({
           type: 'start_tracking',
           person_embed: 'gAAAAABhN2K8...',
           video_source: 0,
           fps_limit: 10
       }));
   };

   ws.onmessage = function(event) {
       const data = JSON.parse(event.data);
       if (data.type === 'matches') {
           console.log('Found matches:', data.matches);
           // Draw bounding boxes
           data.matches.forEach(match => {
               drawBoundingBox(match.bbox, match.distance);
           });
       }
   };

Error Handling
--------------

HTTP Status Codes
~~~~~~~~~~~~~~~~~

- ``200 OK``: Request successful
- ``400 Bad Request``: Invalid request format
- ``422 Unprocessable Entity``: Valid format but processing failed
- ``500 Internal Server Error``: Server error

Error Response Format
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "detail": "Error message describing what went wrong",
     "error_code": "PROCESSING_FAILED",
     "timestamp": "2025-11-02T15:30:00Z"
   }

Common Errors
~~~~~~~~~~~~~

**No Person Detected:**

.. code-block:: json

   {
     "detail": "No person detected, please try with another image",
     "error_code": "NO_PERSON_DETECTED"
   }

**Invalid Image Format:**

.. code-block:: json

   {
     "detail": "Could not decode image file",
     "error_code": "INVALID_IMAGE_FORMAT"
   }

**File Not Found:**

.. code-block:: json

   {
     "detail": "Video file not found at specified path",
     "error_code": "FILE_NOT_FOUND"
   }

Rate Limiting
-------------

WebSocket Connections
~~~~~~~~~~~~~~~~~~~~~

- Maximum 10 concurrent connections per IP
- Frame processing limited to 10 FPS per session
- Automatic disconnection after 5 minutes of inactivity

HTTP Endpoints
~~~~~~~~~~~~~~

- No rate limiting currently implemented

Request Limits
~~~~~~~~~~~~~~

- Maximum image size: 100MB
- Maximum video size: 100MB
- Processing timeout: 300 seconds

Integration Examples
--------------------

Python Client
~~~~~~~~~~~~~

.. code-block:: python

   import requests
   import base64

   # Upload person image
   with open('person.jpg', 'rb') as f:
       image_data = base64.b64encode(f.read()).decode()

   response = requests.post('http://localhost:5000/upload-embedding',
                          json={'image': image_data})

   embedding = response.json()['embedding']
   print(f"Generated embedding: {embedding[:50]}...")

API Documentation
~~~~~~~~~~~~~~~~~

Interactive API documentation is available at:

- Swagger UI: ``http://localhost:5000/docs``

Performance Considerations
--------------------------

Optimization Tips
~~~~~~~~~~~~~~~~~

- Use GPU acceleration when available
- Process videos in smaller chunks for large files
- Implement caching for frequently used embeddings
- Use WebSocket for real-time applications
- Batch multiple images when possible

Monitoring
~~~~~~~~~~

- Check server logs for performance metrics
- Monitor GPU memory usage during processing
- Track WebSocket connection counts
- Monitor processing times for optimization
