WebSocket Examples
==================

This guide demonstrates how to use WebSocket connections for real-time person tracking and video stream processing with the WatchMe AI Backend.

Overview
--------

WebSocket connections enable:

* Real-time person tracking in live video streams
* Bidirectional communication between client and server
* Low-latency frame processing and match detection
* Session management for multiple concurrent users

Connection Setup
----------------

Basic WebSocket Connection
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Endpoint:** ``ws://localhost:5000/video-stream/{session_id}``

**JavaScript Example:**

.. code-block:: javascript

   const sessionId = 'session_' + Date.now();
   const ws = new WebSocket(`ws://localhost:5000/video-stream/${sessionId}`);

   ws.onopen = function() {
       console.log('✓ WebSocket connected');
   };

   ws.onmessage = function(event) {
       const data = JSON.parse(event.data);
       handleMessage(data);
   };

   ws.onerror = function(error) {
       console.error('✗ WebSocket error:', error);
   };

   ws.onclose = function() {
       console.log('WebSocket connection closed');
   };

**Python Example:**

.. code-block:: python

   import asyncio
   import websockets
   import json

   async def connect_websocket():
       session_id = "python_session_123"
       uri = f"ws://localhost:5000/video-stream/{session_id}"

       async with websockets.connect(uri) as websocket:
           print("✓ WebSocket connected")

           # Your WebSocket logic here
           await websocket.send(json.dumps({
               "type": "ping"
           }))

           response = await websocket.recv()
           print(f"Server response: {response}")

   # Run the connection
   asyncio.run(connect_websocket())

Message Types
-------------

Client to Server Messages
~~~~~~~~~~~~~~~~~~~~~~~~~

**Start Tracking:**

.. code-block:: json

   {
     "type": "start_tracking",
     "person_embed": "base64_encrypted_embedding",
     "video_source": 0,
     "fps_limit": 10
   }

**Single Frame Processing:**

.. code-block:: json

   {
     "type": "single_frame",
     "frame": "base64_encoded_frame_data",
     "person_embed": "base64_encrypted_embedding"
   }

**Stop Tracking:**

.. code-block:: json

   {
     "type": "stop_tracking"
   }

**Ping (Health Check):**

.. code-block:: json

   {
     "type": "ping"
   }

Server to Client Messages
~~~~~~~~~~~~~~~~~~~~~~~~~

**Match Results:**

.. code-block:: json

   {
     "type": "matches",
     "matches": [
       {
         "bbox": [100, 150, 200, 400],
         "distance": 0.23,
         "confidence": 0.85
       }
     ],
     "timestamp": 1635789012.34,
     "frame_info": {
       "frame_number": 120,
       "width": 1920,
       "height": 1080,
       "processed_at": 1635789012.34
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

**Pong Response:**

.. code-block:: json

   {
     "type": "status",
     "status": "pong",
     "timestamp": 1635789012.34
   }

Complete Examples
-----------------

Real-time Webcam Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~

**HTML Interface:**

.. code-block:: html

   <!DOCTYPE html>
   <html>
   <head>
       <title>Real-time Person Tracking</title>
   </head>
   <body>
       <div>
           <video id="webcam" width="640" height="480" autoplay></video>
           <canvas id="overlay" width="640" height="480"></canvas>
       </div>

       <div>
           <button onclick="startTracking()">Start Tracking</button>
           <button onclick="stopTracking()">Stop Tracking</button>
           <input type="file" id="personImage" accept="image/*">
       </div>

       <div id="status"></div>

       <script src="websocket-tracker.js"></script>
   </body>
   </html>

**JavaScript Implementation:**

.. code-block:: javascript

   class RealTimeTracker {
       constructor() {
           this.ws = null;
           this.personEmbedding = null;
           this.video = document.getElementById('webcam');
           this.canvas = document.getElementById('overlay');
           this.ctx = this.canvas.getContext('2d');
           this.isTracking = false;

           this.setupWebcam();
           this.setupFileUpload();
       }

       async setupWebcam() {
           try {
               const stream = await navigator.mediaDevices.getUserMedia({
                   video: true
               });
               this.video.srcObject = stream;
               console.log('✓ Webcam access granted');
           } catch (error) {
               console.error('✗ Webcam access denied:', error);
               this.updateStatus('Error: Webcam access required');
           }
       }

       setupFileUpload() {
           const fileInput = document.getElementById('personImage');
           fileInput.addEventListener('change', async (event) => {
               const file = event.target.files[0];
               if (file) {
                   await this.uploadPersonImage(file);
               }
           });
       }

       async uploadPersonImage(file) {
           try {
               // Convert file to base64
               const base64 = await this.fileToBase64(file);

               // Upload to AI backend
               const response = await fetch('http://localhost:5000/upload-embedding', {
                   method: 'POST',
                   headers: {
                       'Content-Type': 'application/json',
                   },
                   body: JSON.stringify({ image: base64 })
               });

               if (response.ok) {
                   const result = await response.json();
                   this.personEmbedding = result.embedding;
                   this.updateStatus('✓ Person embedding created');
                   console.log('Person embedding ready');
               } else {
                   throw new Error('Failed to create embedding');
               }
           } catch (error) {
               console.error('Error uploading image:', error);
               this.updateStatus('✗ Failed to upload image');
           }
       }

       fileToBase64(file) {
           return new Promise((resolve, reject) => {
               const reader = new FileReader();
               reader.readAsDataURL(file);
               reader.onload = () => {
                   const base64 = reader.result.split(',')[1];
                   resolve(base64);
               };
               reader.onerror = error => reject(error);
           });
       }

       async connectWebSocket() {
           const sessionId = 'webcam_session_' + Date.now();
           this.ws = new WebSocket(`ws://localhost:5000/video-stream/${sessionId}`);

           return new Promise((resolve, reject) => {
               this.ws.onopen = () => {
                   console.log('✓ WebSocket connected');
                   this.updateStatus('Connected to tracking service');
                   resolve();
               };

               this.ws.onmessage = (event) => {
                   const data = JSON.parse(event.data);
                   this.handleWebSocketMessage(data);
               };

               this.ws.onerror = (error) => {
                   console.error('✗ WebSocket error:', error);
                   reject(error);
               };

               this.ws.onclose = () => {
                   console.log('WebSocket disconnected');
                   this.updateStatus('Disconnected from tracking service');
                   this.isTracking = false;
               };
           });
       }

       handleWebSocketMessage(data) {
           switch (data.type) {
               case 'matches':
                   this.drawMatches(data.matches, data.frame_info);
                   break;
               case 'status':
                   console.log(`Status: ${data.status} - ${data.message}`);
                   this.updateStatus(data.message);
                   break;
           }
       }

       drawMatches(matches, frameInfo) {
           // Clear previous drawings
           this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

           if (matches.length === 0) return;

           matches.forEach((match, index) => {
               const [x1, y1, x2, y2] = match.bbox;
               const distance = match.distance;

               // Scale coordinates to canvas size
               const scaleX = this.canvas.width / (frameInfo.width || this.canvas.width);
               const scaleY = this.canvas.height / (frameInfo.height || this.canvas.height);

               const scaledX1 = x1 * scaleX;
               const scaledY1 = y1 * scaleY;
               const scaledX2 = x2 * scaleX;
               const scaledY2 = y2 * scaleY;

               // Draw bounding box
               this.ctx.strokeStyle = distance < 0.5 ? '#00ff00' : '#ffff00';
               this.ctx.lineWidth = 3;
               this.ctx.strokeRect(
                   scaledX1, scaledY1,
                   scaledX2 - scaledX1, scaledY2 - scaledY1
               );

               // Draw distance text
               this.ctx.fillStyle = distance < 0.5 ? '#00ff00' : '#ffff00';
               this.ctx.font = 'bold 16px Arial';
               this.ctx.fillText(
                   `Match ${index + 1}: ${distance.toFixed(3)}`,
                   scaledX1, scaledY1 - 10
               );
           });
       }

       async startTracking() {
           if (!this.personEmbedding) {
               alert('Please upload a person image first');
               return;
           }

           try {
               await this.connectWebSocket();

               // Start live tracking
               this.ws.send(JSON.stringify({
                   type: 'start_tracking',
                   person_embed: this.personEmbedding,
                   video_source: 0,  // Webcam
                   fps_limit: 5     // Process 5 frames per second
               }));

               this.isTracking = true;
               this.updateStatus('🔍 Tracking started');

               // Send frames periodically
               this.frameInterval = setInterval(() => {
                   if (this.isTracking && this.video.readyState >= 2) {
                       this.sendCurrentFrame();
                   }
               }, 1000); // Send frame every second

           } catch (error) {
               console.error('Failed to start tracking:', error);
               this.updateStatus('✗ Failed to start tracking');
           }
       }

       sendCurrentFrame() {
           // Create temporary canvas to capture frame
           const tempCanvas = document.createElement('canvas');
           const tempCtx = tempCanvas.getContext('2d');

           tempCanvas.width = this.video.videoWidth;
           tempCanvas.height = this.video.videoHeight;

           tempCtx.drawImage(this.video, 0, 0);

           // Convert to base64
           const frameData = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];

           // Send frame via WebSocket
           if (this.ws && this.ws.readyState === WebSocket.OPEN) {
               this.ws.send(JSON.stringify({
                   type: 'single_frame',
                   frame: frameData,
                   person_embed: this.personEmbedding
               }));
           }
       }

       stopTracking() {
           if (this.ws && this.ws.readyState === WebSocket.OPEN) {
               this.ws.send(JSON.stringify({
                   type: 'stop_tracking'
               }));
           }

           if (this.frameInterval) {
               clearInterval(this.frameInterval);
           }

           if (this.ws) {
               this.ws.close();
           }

           this.isTracking = false;
           this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
           this.updateStatus('Tracking stopped');
       }

       updateStatus(message) {
           document.getElementById('status').textContent = message;
       }
   }

   // Global functions for buttons
   let tracker;

   window.onload = function() {
       tracker = new RealTimeTracker();
   };

   function startTracking() {
       tracker.startTracking();
   }

   function stopTracking() {
       tracker.stopTracking();
   }

Python Async Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   import websockets
   import json
   import cv2
   import base64
   import time

   class AsyncPersonTracker:
       def __init__(self, ws_url="ws://localhost:5000"):
           self.ws_url = ws_url
           self.websocket = None
           self.person_embedding = None
           self.is_tracking = False

       async def connect(self, session_id):
           """Connect to WebSocket server"""
           try:
               uri = f"{self.ws_url}/video-stream/{session_id}"
               self.websocket = await websockets.connect(uri)
               print(f"✓ Connected to {uri}")
               return True
           except Exception as e:
               print(f"✗ Connection failed: {e}")
               return False

       async def upload_person_image(self, image_path):
           """Upload person image and get embedding"""
           import aiohttp

           with open(image_path, 'rb') as f:
               image_data = base64.b64encode(f.read()).decode()

           async with aiohttp.ClientSession() as session:
               async with session.post(
                   'http://localhost:5000/upload-embedding',
                   json={'image': image_data}
               ) as response:
                   if response.status == 200:
                       result = await response.json()
                       self.person_embedding = result['embedding']
                       print("✓ Person embedding created")
                       return True
                   else:
                       print(f"✗ Failed to create embedding: {response.status}")
                       return False

       async def start_live_tracking(self, video_source=0):
           """Start live video tracking"""
           if not self.person_embedding:
               print("✗ No person embedding available")
               return

           # Start tracking on server
           await self.websocket.send(json.dumps({
               "type": "start_tracking",
               "person_embed": self.person_embedding,
               "video_source": video_source,
               "fps_limit": 10
           }))

           self.is_tracking = True
           print("✓ Live tracking started")

       async def send_frame_from_video(self, video_source=0):
           """Capture and send frames from video source"""
           cap = cv2.VideoCapture(video_source)

           if not cap.isOpened():
               print("✗ Cannot open video source")
               return

           try:
               while self.is_tracking:
                   ret, frame = cap.read()
                   if not ret:
                       break

                   # Encode frame to base64
                   _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                   frame_b64 = base64.b64encode(buffer).decode()

                   # Send frame
                   await self.websocket.send(json.dumps({
                       "type": "single_frame",
                       "frame": frame_b64,
                       "person_embed": self.person_embedding
                   }))

                   # Control frame rate
                   await asyncio.sleep(0.5)  # 2 FPS

           finally:
               cap.release()

       async def listen_for_matches(self):
           """Listen for match results from server"""
           try:
               while self.is_tracking:
                   message = await self.websocket.recv()
                   data = json.loads(message)

                   if data["type"] == "matches":
                       matches = data["matches"]
                       frame_info = data.get("frame_info", {})

                       print(f"📍 Found {len(matches)} matches in frame {frame_info.get('frame_number', '?')}")

                       for i, match in enumerate(matches):
                           bbox = match["bbox"]
                           distance = match["distance"]
                           print(f"  Match {i+1}: bbox={bbox}, distance={distance:.3f}")

                   elif data["type"] == "status":
                       print(f"ℹ️  Status: {data['status']} - {data['message']}")

           except websockets.exceptions.ConnectionClosed:
               print("WebSocket connection closed")

       async def stop_tracking(self):
           """Stop tracking and close connection"""
           if self.websocket:
               await self.websocket.send(json.dumps({"type": "stop_tracking"}))
               self.is_tracking = False
               await self.websocket.close()
               print("✓ Tracking stopped")

       async def run_tracking_session(self, person_image_path, video_source=0):
           """Complete tracking session"""
           session_id = f"python_session_{int(time.time())}"

           # Connect to WebSocket
           if not await self.connect(session_id):
               return

           # Upload person image
           if not await self.upload_person_image(person_image_path):
               return

           try:
               # Start tracking
               await self.start_live_tracking(video_source)

               # Create tasks for frame sending and match listening
               frame_task = asyncio.create_task(self.send_frame_from_video(video_source))
               listen_task = asyncio.create_task(self.listen_for_matches())

               # Run both tasks concurrently
               await asyncio.gather(frame_task, listen_task)

           except KeyboardInterrupt:
               print("\nStopping tracking session...")

           finally:
               await self.stop_tracking()

   # Usage example
   async def main():
       tracker = AsyncPersonTracker()

       # Run tracking session with person image and webcam
       await tracker.run_tracking_session(
           person_image_path="reference_person.jpg",
           video_source=0  # Webcam
       )

   if __name__ == "__main__":
       # Install required packages: pip install aiohttp opencv-python websockets
       print("Starting async person tracker...")
       print("Press Ctrl+C to stop")
       asyncio.run(main())

Error Handling and Reconnection
-------------------------------

Robust WebSocket Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: javascript

   class RobustWebSocketClient {
       constructor(baseUrl = 'ws://localhost:5000') {
           this.baseUrl = baseUrl;
           this.ws = null;
           this.sessionId = null;
           this.reconnectAttempts = 0;
           this.maxReconnectAttempts = 5;
           this.reconnectDelay = 1000; // Start with 1 second
           this.isConnected = false;
           this.messageQueue = [];
       }

       connect(sessionId) {
           this.sessionId = sessionId;

           return new Promise((resolve, reject) => {
               const wsUrl = `${this.baseUrl}/video-stream/${sessionId}`;
               console.log(`Connecting to ${wsUrl}...`);

               this.ws = new WebSocket(wsUrl);

               this.ws.onopen = () => {
                   console.log('✓ WebSocket connected');
                   this.isConnected = true;
                   this.reconnectAttempts = 0;
                   this.reconnectDelay = 1000;

                   // Send queued messages
                   this.flushMessageQueue();
                   resolve();
               };

               this.ws.onmessage = (event) => {
                   try {
                       const data = JSON.parse(event.data);
                       this.handleMessage(data);
                   } catch (error) {
                       console.error('Error parsing WebSocket message:', error);
                   }
               };

               this.ws.onerror = (error) => {
                   console.error('WebSocket error:', error);
                   this.isConnected = false;
               };

               this.ws.onclose = (event) => {
                   console.log(`WebSocket closed: ${event.code} - ${event.reason}`);
                   this.isConnected = false;

                   // Attempt to reconnect
                   this.attemptReconnect();
               };

               // Connection timeout
               setTimeout(() => {
                   if (!this.isConnected) {
                       reject(new Error('Connection timeout'));
                   }
               }, 10000);
           });
       }

       attemptReconnect() {
           if (this.reconnectAttempts >= this.maxReconnectAttempts) {
               console.error('Max reconnection attempts reached');
               return;
           }

           this.reconnectAttempts++;
           console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

           setTimeout(() => {
               this.connect(this.sessionId).catch(error => {
                   console.error('Reconnection failed:', error);
                   this.reconnectDelay *= 2; // Exponential backoff
               });
           }, this.reconnectDelay);
       }

       send(message) {
           if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
               this.ws.send(JSON.stringify(message));
           } else {
               // Queue message for later sending
               this.messageQueue.push(message);
               console.log('Message queued (not connected)');
           }
       }

       flushMessageQueue() {
           while (this.messageQueue.length > 0 && this.isConnected) {
               const message = this.messageQueue.shift();
               this.ws.send(JSON.stringify(message));
           }
       }

       handleMessage(data) {
           // Override this method in your implementation
           console.log('Received message:', data);
       }

       close() {
           this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
           if (this.ws) {
               this.ws.close();
           }
       }
   }

Performance Optimization
------------------------

Frame Rate Control
~~~~~~~~~~~~~~~~~~

.. code-block:: javascript

   class OptimizedTracker extends RobustWebSocketClient {
       constructor() {
           super();
           this.lastFrameTime = 0;
           this.targetFPS = 5; // Process 5 frames per second
           this.frameInterval = 1000 / this.targetFPS;
           this.frameQueue = [];
           this.maxQueueSize = 3;
       }

       sendFrame(frameData, embedding) {
           const now = Date.now();

           // Rate limiting
           if (now - this.lastFrameTime < this.frameInterval) {
               return; // Skip this frame
           }

           // Queue management
           if (this.frameQueue.length >= this.maxQueueSize) {
               this.frameQueue.shift(); // Remove oldest frame
           }

           this.frameQueue.push({
               frame: frameData,
               embedding: embedding,
               timestamp: now
           });

           this.processFrameQueue();
           this.lastFrameTime = now;
       }

       processFrameQueue() {
           if (this.frameQueue.length === 0 || !this.isConnected) {
               return;
           }

           const frameData = this.frameQueue.shift();

           this.send({
               type: 'single_frame',
               frame: frameData.frame,
               person_embed: frameData.embedding
           });
       }
   }

Best Practices
--------------

Security Considerations
~~~~~~~~~~~~~~~~~~~~~~~

1. **Validate session IDs** on the server side
2. **Implement rate limiting** to prevent abuse
3. **Use HTTPS/WSS** in production
4. **Sanitize all input data** before processing

Resource Management
~~~~~~~~~~~~~~~~~~~

1. **Always close WebSocket connections** when done
2. **Clear intervals and timeouts** to prevent memory leaks
3. **Limit frame processing rate** to prevent server overload
4. **Implement connection pooling** for multiple sessions

Monitoring and Debugging
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: javascript

   class MonitoredWebSocketClient extends RobustWebSocketClient {
       constructor() {
           super();
           this.stats = {
               messagesSent: 0,
               messagesReceived: 0,
               connectionsEstablished: 0,
               reconnections: 0,
               errors: 0
           };
       }

       send(message) {
           super.send(message);
           this.stats.messagesSent++;
           this.logStats();
       }

       handleMessage(data) {
           super.handleMessage(data);
           this.stats.messagesReceived++;

           if (data.type === 'matches') {
               console.log(`📊 Processing time: ${data.frame_info?.processing_time || 'unknown'}`);
           }
       }

       logStats() {
           if (this.stats.messagesSent % 10 === 0) { // Log every 10 messages
               console.log('📈 WebSocket Stats:', this.stats);
           }
       }
   }

This comprehensive WebSocket documentation covers all aspects of real-time communication with the WatchMe AI Backend, from basic connections to advanced error handling and optimization techniques.
