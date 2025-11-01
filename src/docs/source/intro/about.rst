About WatchMe AI
================

Overview
--------

WatchMe AI Backend is a person re-identification system designed for:

* Real-time person tracking in video streams
* Batch video processing for person search
* Secure embedding storage with AES encryption
* REST and WebSocket API endpoints

Key Features
------------

Person Re-identification
~~~~~~~~~~~~~~~~~~~~~~~

Uses OSNet (Omni-Scale Network) for generating robust person embeddings:

* High accuracy on standard benchmarks
* Real-time inference capability
* Support for various input formats

Object Detection
~~~~~~~~~~~~~~~

YOLO (You Only Look Once) integration for:

* Fast person detection in images/videos
* Bounding box extraction
* Batch processing support

Security
~~~~~~~~

* AES encryption for person embeddings
* Secure API endpoints
* Environment-based configuration

Architecture
-----------

The system follows a clean architecture pattern:

.. code-block:: text

   ┌─────────────────┐    ┌─────────────────┐
   │   FastAPI       │    │   WebSocket     │
   │   Endpoints     │    │   Protocol      │
   └─────────────────┘    └─────────────────┘
            │                       │
   ┌─────────────────────────────────────────┐
   │           Application Layer             │
   │  ┌─────────────────┐ ┌─────────────────┐│
   │  │ Create Embedding│ │ Predict Person  ││
   │  └─────────────────┘ └─────────────────┘│
   └─────────────────────────────────────────┘
            │                       │
   ┌─────────────────────────────────────────┐
   │          Infrastructure Layer           │
   │  ┌─────────────────┐ ┌─────────────────┐│
   │  │      YOLO       │ │     OSNet       ││
   │  │   Detection     │ │   Encoding      ││
   │  └─────────────────┘ └─────────────────┘│
   └─────────────────────────────────────────┘
