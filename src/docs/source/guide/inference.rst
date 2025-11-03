Inference Guide
===============

This guide covers how to perform inference with trained models in the WatchMe AI Backend, including person detection with YOLO and person re-identification with OSNet.

Overview
--------

The WatchMe AI Backend supports two main types of inference:

* **YOLO Inference**: Detect and locate people in images/videos
* **OSNet Inference**: Generate embeddings for person re-identification
* **Combined Pipeline**: Full person search workflow

The system is designed for both batch processing and real-time inference scenarios.

YOLO Inference
--------------

Bounding Box Detection
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # python -m src.infrastructure.yolo.core;predict.py
    def predict(images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]) -> List[Dict[str, Any]]:
        """
        Runs object detection and returns both bounding boxes and cropped person images.

        Args:
            images: Single image path/array or list of image paths/arrays to process.

        Returns:
            List of dictionaries containing detection results for each image.
            Returns empty list if no detections found.

        Raises:
            RuntimeError: If YOLO prediction fails.
            ValueError: If input images are invalid.
        """
        settings = YOLOSettings()
        model = yolo_client(settings.YOLO_MODEL_PATH) #Or YOLO(settings.YOLO_MODEL_PATH) to load the model

        if not isinstance(images, list):
            images = [images]

        try:
            results = model.predict(
                images,
                stream=True,
                conf=0.28,
                classes=[0],
                verbose=False,
            )

            results_list = list(results)

            if not results_list:
                return []

            bounding_boxes = get_bounding_boxes(images, results_list)
            return bounding_boxes if bounding_boxes else []

        except Exception as e:
            raise RuntimeError(f"YOLO prediction failed: {str(e)}") from e


OSNet Inference
---------------

Load Model
~~~~~~~~~~

.. code-block:: python

    # python -m src.infrastructure.osnet.script.load_checkpoint

    def load_checkpoint(weights_path, device, model):
        """
        Load a pre-trained model checkpoint and prepare it for inference.
        Args:
            weights_path (str): Path to the checkpoint file containing the model weights.
            device (torch.device): Device to load the model onto (e.g., 'cpu' or 'cuda').
            model (torch.nn.Module): The model instance to load the weights into.
        Returns:
            torch.nn.Module: The model with loaded weights, moved to the specified device
                            and set to evaluation mode.
        Note:
            This function assumes the checkpoint is in Torchreid format with weights
            stored under the 'state_dict' key. Modify accordingly for other formats.
            Uses strict=False when loading state dict to allow partial loading.
        """
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']

        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith('classifier') and
                         not k.endswith('running_mean') and
                         not k.endswith('running_var')}

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model

Person Embedding
~~~~~~~~~~~~~~~~

.. code-block:: python

    # python -m src.infrastructure.osnet.core.encode
    class OSNetEncoder:
        """Handle OSNet encoding operations for person re-identification."""

        def __init__(self):
            self.osnet_client = OSNetModel()
            self.settings = OSNetSettings()
            self.model = self.osnet_client.create_osnet_model(
                num_classes=self.settings.OSNET_NUM_CLASSES
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.weights_path = Path(self.settings.OSNET_SAVE_DIR, self.settings.OSNET_MODEL_NAME)
            self.transform = create_transforms(self.settings.OSNET_IMG_HEIGHT,
                                               self.settings.OSNET_IMG_WIDTH)
            self._load_model()

        def _load_model(self):
            """Load the OSNet model with pre-trained weights."""
            self.model = load_checkpoint(self.weights_path, self.device, self.model)
            print("OSNet model loaded successfully")

        def encode_single_image(self, image):
            """
            Encode a single image to feature vector.

            Args:
                image: Input image (PIL Image, numpy array, or tensor)

            Returns:
                numpy.ndarray: Feature vector (1D array)
            """

            image_tensor = preprocess_image(image, self.transform)
            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                features = self.model(image_tensor)

                if isinstance(features, tuple):
                    features = features[0]

                features = torch.nn.functional.normalize(features, p=2, dim=1)
                features = features.cpu().numpy().flatten()

            return features

        def encode_batch(self, images):
            """
            Encode a batch of images to feature vectors.

            Args:
                images: List of images (PIL Images, numpy arrays, or tensors)

            Returns:
                numpy.ndarray: Feature matrix (num_images x feature_dim)
            """
            if not images:
                return np.array([])

            image_tensors = []
            for image in images:
                tensor = preprocess_image(image, self.transform)
                image_tensors.append(tensor.squeeze(0))

            batch_tensor = torch.stack(image_tensors).to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor)

                if isinstance(features, tuple):
                    features = features[0]

                features = torch.nn.functional.normalize(features, p=2, dim=1)
                features = features.cpu().numpy()

            return features


Combined Pipeline
-----------------

Complete Person Search
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    encoder = OSNetEncoder()

    def predict_person_on_stream(chosen_person, stream):
        """
        Compare the chosen person's embedding to all detected people in the video stream.
        Args:
            chosen_person: Encrypted embedding of the target person.
            stream: Video frame(s) or stream to process.
        Returns:
            List of matching bounding boxes.
        """
        people_results = predict(stream)
        all_cropped_images = []
        all_bboxes = []

        for frame_result in people_results:
            for detection in frame_result['detections']:
                all_cropped_images.append(detection['cropped_image'])
                all_bboxes.append(detection['bbox'])

        if not all_cropped_images:
            return []

        decrypted_embedding = decrypt_embedding(chosen_person, shape=(512,), dtype='float32')

        encoded_batch = encoder.encode_batch(all_cropped_images)
        matches = []

        for i, encoded_person in enumerate(encoded_batch):
            distance = compute_euclidean_distance(decrypted_embedding, encoded_person)
            if distance < 0.8:
                matches.append({
                    "bbox": all_bboxes[i],
                    "distance": distance
                })

        return matches
