About WatchMe
==============

Overview
--------

WatchMe is a comprehensive person re-identification and tracking system consisting of three interconnected entities working together to achieve end-to-end person tracking through live feeds and video processing.

System Components
~~~~~~~~~~~~~~~~~

- `WatchMe UI <https://github.com/beatrizamante/watchme_frontend>`__ - Frontend application for user interaction
- `WatchMe API <https://github.com/beatrizamante/watchme_api>`__ - Backend API for business logic and data management
- `WatchMe AI Backend <https://github.com/beatrizamante/watchme_ai>`__ - AI engine for person detection and re-identification

Each component has its own setup, dependencies, and specialized functionality that contributes to the overall system capabilities.

WatchMe AI Backend Features
---------------------------

The AI Backend serves as the core engine for person re-identification and is designed for:

* Real-time person tracking in video streams
* Batch video processing for person search
* Secure embedding storage with AES encryption
* REST and WebSocket API endpoints for real-time communication

Person Re-identification
~~~~~~~~~~~~~~~~~~~~~~~~

Uses OSNet (Omni-Scale Network) for generating robust person embeddings:

* High accuracy on standard benchmarks (Market1501, DukeMTMC)
* Real-time inference capability with GPU acceleration
* Support for various input formats (images, video streams, webcam)
* Encrypted embedding storage for security

Object Detection
~~~~~~~~~~~~~~~~

YOLO (You Only Look Once) integration for:

* Fast person detection in images and videos
* Accurate bounding box extraction and cropping
* Batch processing support for multiple detections

Security & Performance
~~~~~~~~~~~~~~~~~~~~~~

* AES encryption for person embeddings
* Environment-based configuration management
* Optimized for both CPU and GPU inference

WatchMe API Features
--------------------

The TypeScript API (`WatchMe API <https://github.com/beatrizamante/watchme_api>`__) provides comprehensive backend services including:

Authentication & Security
~~~~~~~~~~~~~~~~~~~~~~~~~

* JWT cookie-based authentication system
* Secure HTTP token handling
* Protected route middleware
* Session management and logout functionality
* Websocket proxy for person search

User Management
~~~~~~~~~~~~~~~

**Administrative Features:**

* Create user accounts (``POST /user``)
* Retrieve user information (``GET /user?id=:userId``)
* List all users with filtering (``GET /users?active=:boolean``)
* Delete person embeddings (``DELETE /person?id=:personId``)

**User Features:**

* User registration (``POST /register``)
* User authentication (``POST /login``)
* Profile management (``PATCH /user?id=:userId``)
* Profile picture management (``DELETE /user/picture?id=:userId``)
* Secure logout (``POST /logout``)

Person & Video Management
~~~~~~~~~~~~~~~~~~~~~~~~~

**Person Hash Management:**

* List all person embeddings (``GET /people``)
* Retrieve specific person data (``GET /person?id=:personId``)
* Create person embeddings (``POST /person``)
* Search for people in videos (``GET /person/find?id=:personId&videoId=:videoId``)

**Video Processing:**

* Upload videos for processing (``POST /video``)
* List user's videos (``GET /videos``)
* Retrieve specific video (``GET /videos?id=:videoId``)
* Delete user videos (``DELETE /video?id=:videoId``)

Data Persistence
~~~~~~~~~~~~~~~~

* Secure file storage for videos and images
* Database management for user data and embeddings
* Efficient video metadata handling
* Integration with AI backend for processing

WatchMe UI Features
-------------------

The Frontend application (`WatchMe UI <https://github.com/beatrizamante/watchme_frontend>`__) provides an intuitive user interface featuring:

User Interface
~~~~~~~~~~~~~~

* Clean, responsive design for all device types
* Intuitive file upload interface for images and videos

File Management
~~~~~~~~~~~~~~~

* Drag-and-drop file upload functionality
* Support for multiple image and video formats
* File size validation and error handling

User Experience
~~~~~~~~~~~~~~~

* Secure user authentication and registration flows
* User profile management interface
* Dashboard for managing uploaded content
* Real-time notifications and status updates

Visualization
~~~~~~~~~~~~~

* Video playback with bounding box overlays
* Person tracking visualization in real-time
* Results display with confidence scores
* Export capabilities for processed results

System Architecture
-------------------

The complete WatchMe system follows a microservices architecture pattern:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    WatchMe Frontend                         │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
   │  │   File Upload   │  │  User Interface │  │ Video Player│  │
   │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
   └─────────────────────────────────────────────────────────────┘
                                   │
                                   │ HTTP/WebSocket
                                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                    WatchMe API                              │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
   │  │ Authentication  │  │ User Management │  │File Storage │  │
   │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
   └─────────────────────────────────────────────────────────────┘
                                   │
                                   │ HTTP/WebSocket
                                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                 WatchMe AI Backend                          │
   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
   │  │      YOLO       │  │     OSNet       │  │  WebSocket  │  │
   │  │   Detection     │  │   Encoding      │  │  Protocol   │  │
   │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
   └─────────────────────────────────────────────────────────────┘

Data Flow
~~~~~~~~~

1. **User Interaction**: Users interact with the frontend to upload images/videos
2. **API Processing**: The TypeScript API handles authentication, file storage, and business logic
3. **AI Processing**: The AI backend performs person detection and re-identification
4. **Real-time Updates**: WebSocket connections enable real-time tracking and notifications
5. **Results Display**: Processed results are displayed in the frontend with visual overlays

Integration Points
~~~~~~~~~~~~~~~~~~

* **Frontend ↔ API**: RESTful HTTP communication and WebSocket for real-time updates
* **API ↔ AI Backend**: HTTP requests for batch processing and WebSocket for live streaming
* **Shared Authentication**: JWT tokens passed through the entire system
* **File Management**: Coordinated file storage and retrieval across components

Getting Started
---------------

To set up the complete WatchMe system:

1. **AI Backend**: Follow the :doc:`../usage/configuration` guide for this repository
2. **TypeScript API**: See the `WatchMe API documentation <https://github.com/beatrizamante/watchme_api>`__
3. **Frontend**: Refer to the `WatchMe UI setup guide <https://github.com/beatrizamante/watchme_frontend>`__

Each component can be developed and deployed independently, making the system highly modular and scalable.
