# Add to your test_routes.py
from fastapi import APIRouter
import numpy as np
from src.infrastructure.osnet.core.encode import OSNetEncoder, get_encoder
from src.infrastructure.yolo.core.predict import predict

router = APIRouter()

@router.post("/test-single-vs-batch-encoding")
async def test_single_vs_batch_encoding():
    """Compare single vs batch encoding on the same image"""

    try:
        # Use your test image
        image_path = "C:/Users/beatr/Downloads/test_new.png"

        # Get detection using same method as creation
        people_results = predict(image_path)
        if not people_results or not people_results[0]['detections']:
            return {"error": "No person detected"}

        cropped_image = people_results[0]['detections'][0]['cropped_image']

        # Test both encoding methods
        encoder = get_encoder()

        # Method 1: Single image (like creation)
        embedding_single = encoder.encode_single_image(cropped_image)

        # Method 2: Batch with one image (like prediction)
        embedding_batch = encoder.encode_batch([cropped_image])[0]

        # Compare
        distance_between_methods = float(np.linalg.norm(embedding_single - embedding_batch))

        return {
            "test_results": {
                "same_image_different_methods": True,
                "distance_between_methods": distance_between_methods,
                "should_be_zero": distance_between_methods < 1e-6
            },
            "embedding_analysis": {
                "single_method": {
                    "shape": list(embedding_single.shape),
                    "norm": float(np.linalg.norm(embedding_single)),
                    "mean": float(embedding_single.mean()),
                    "std": float(embedding_single.std())
                },
                "batch_method": {
                    "shape": list(embedding_batch.shape),
                    "norm": float(np.linalg.norm(embedding_batch)),
                    "mean": float(embedding_batch.mean()),
                    "std": float(embedding_batch.std())
                }
            },
            "preprocessing_comparison": {
                "single_uses": "preprocess_image() -> single tensor",
                "batch_uses": "preprocess_image() for each -> stack tensors",
                "potential_difference": "Tensor stacking or batch normalization"
            },
            "recommendation": {
                "issue_found": bool(distance_between_methods > 1e-3),
                "fix_needed": "Use same encoding method for both creation and prediction"
            }
        }

    except Exception as e:
        return {"error": str(e)}

@router.post("/test-full-pipeline-consistency")
async def test_full_pipeline_consistency():
    """Test the full creation vs prediction pipeline"""

    try:
        image_path = "C:/Users/beatr/Downloads/test_new.png"

        # Step 1: Create embedding exactly like your creation pipeline
        from src.application.use_cases.create_person_embedding import create_person_embedding
        encrypted_embedding = create_person_embedding(image_path)

        # Step 2: Decrypt it (like prediction does)
        from src._lib.decrypt import decrypt_embedding
        decrypted_embedding = decrypt_embedding(encrypted_embedding, shape=(512,), dtype='float32')

        # Step 3: Get the same detection again
        people_results = predict(image_path)
        cropped_image = people_results[0]['detections'][0]['cropped_image']

        # Step 4: Encode using prediction method (batch)
        encoder = get_encoder()
        encoded_batch = encoder.encode_batch([cropped_image])
        prediction_embedding = encoded_batch[0]

        # Step 5: Compare
        from src.scripts.calculate_distance import compute_euclidean_distance
        distance = compute_euclidean_distance(decrypted_embedding, prediction_embedding)

        return {
            "pipeline_test": {
                "same_image_full_pipeline": True,
                "creation_to_prediction_distance": float(distance),
                "should_be_very_small": distance < 0.1,
                "is_problematic": distance > 0.5
            },
            "pipeline_flow": {
                "creation": "image -> YOLO -> crop -> encode_single_image -> encrypt",
                "prediction": "image -> YOLO -> crop -> encode_batch -> compare",
                "difference": "encode_single_image vs encode_batch"
            },
            "embeddings_analysis": {
                "decrypted_creation": {
                    "norm": float(np.linalg.norm(decrypted_embedding)),
                    "mean": float(decrypted_embedding.mean())
                },
                "prediction_batch": {
                    "norm": float(np.linalg.norm(prediction_embedding)),
                    "mean": float(prediction_embedding.mean())
                }
            },
            "recommendation": {
                "fix_needed": bool(distance > 0.1),
                "solution": "Use consistent encoding method in both pipelines"
            }
        }

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

@router.post("/debug-pipeline-step-by-step")
async def debug_pipeline_step_by_step():
    """Debug each step of creation vs prediction pipeline"""

    try:
        image_path = "C:/Users/beatr/Downloads/test_new.png"

        # Step 1: Test YOLO consistency - call predict twice
        people_results_1 = predict(image_path)
        people_results_2 = predict(image_path)

        if not (people_results_1 and people_results_2 and
                people_results_1[0]['detections'] and people_results_2[0]['detections']):
            return {"error": "YOLO detection failed"}

        crop_1 = people_results_1[0]['detections'][0]['cropped_image']
        crop_2 = people_results_2[0]['detections'][0]['cropped_image']
        bbox_1 = people_results_1[0]['detections'][0]['bbox']
        bbox_2 = people_results_2[0]['detections'][0]['bbox']

        # Test if YOLO gives identical crops
        crops_identical = np.array_equal(crop_1, crop_2)
        bboxes_identical = np.array_equal(bbox_1, bbox_2)

        # Step 2: Test encoding on identical crops
        encoder = get_encoder()
        embedding_1 = encoder.encode_single_image(crop_1)
        embedding_2 = encoder.encode_single_image(crop_2)

        encoding_distance = float(np.linalg.norm(embedding_1 - embedding_2))

        # Step 3: Test encryption/decryption
        from src._lib.encrypt import encrypt_embedding
        from src._lib.decrypt import decrypt_embedding

        encrypted = encrypt_embedding(embedding_1)
        decrypted = decrypt_embedding(encrypted, shape=(512,), dtype='float32')

        encryption_distance = float(np.linalg.norm(embedding_1 - decrypted))

        # Step 4: Test if different encoder instances matter
        encoder_2 = get_encoder()
        embedding_different_instance = encoder_2.encode_single_image(crop_1)
        instance_distance = float(np.linalg.norm(embedding_1 - embedding_different_instance))

        return {
            "yolo_consistency": {
                "crops_identical": crops_identical,
                "bboxes_identical": bboxes_identical,
                "bbox_1": bbox_1,
                "bbox_2": bbox_2,
                "crop_shapes": [list(crop_1.shape), list(crop_2.shape)]
            },
            "encoding_consistency": {
                "identical_crops_distance": encoding_distance,
                "should_be_zero": encoding_distance < 1e-6
            },
            "encryption_test": {
                "original_vs_decrypted_distance": encryption_distance,
                "encryption_working": encryption_distance < 1e-6
            },
            "encoder_instances": {
                "different_instances_distance": instance_distance,
                "instances_consistent": instance_distance < 1e-6
            },
            "diagnosis": {
                "yolo_issue": not (crops_identical and bboxes_identical),
                "encoding_issue": encoding_distance > 1e-3,
                "encryption_issue": encryption_distance > 1e-3,
                "instance_issue": instance_distance > 1e-3
            }
        }

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

@router.post("/debug-creation-vs-prediction-crops")
async def debug_creation_vs_prediction_crops():
    """Compare the actual crops from creation vs prediction"""

    try:
        image_path = "C:/Users/beatr/Downloads/test_new.png"

        # Step 1: Get crop exactly like creation does
        from src.infrastructure.yolo.core.predict import predict
        creation_results = predict(image_path)  # Same as create_person_embedding calls
        creation_crop = creation_results[0]['detections'][0]['cropped_image']
        creation_bbox = creation_results[0]['detections'][0]['bbox']

        # Step 2: Get crop exactly like prediction does
        # (prediction also calls predict() but might use different path)
        prediction_results = predict(image_path)  # Same call!
        prediction_crop = prediction_results[0]['detections'][0]['cropped_image']
        prediction_bbox = prediction_results[0]['detections'][0]['bbox']

        # Step 3: Compare crops pixel by pixel
        crops_identical = np.array_equal(creation_crop, prediction_crop)
        bboxes_identical = np.array_equal(creation_bbox, prediction_bbox)

        if not crops_identical:
            pixel_diff = np.abs(creation_crop.astype(float) - prediction_crop.astype(float))
            max_pixel_diff = float(pixel_diff.max())
            mean_pixel_diff = float(pixel_diff.mean())
        else:
            max_pixel_diff = 0.0
            mean_pixel_diff = 0.0

        # Step 4: Encode both crops with same encoder
        encoder = get_encoder()
        creation_embedding = encoder.encode_single_image(creation_crop)
        prediction_embedding = encoder.encode_single_image(prediction_crop)

        crop_encoding_distance = float(np.linalg.norm(creation_embedding - prediction_embedding))

        # Step 5: Save crops for visual inspection
        import cv2
        cv2.imwrite("debug_creation_crop.jpg", creation_crop)
        cv2.imwrite("debug_prediction_crop.jpg", prediction_crop)

        return {
            "crop_comparison": {
                "crops_identical": crops_identical,
                "bboxes_identical": bboxes_identical,
                "creation_bbox": creation_bbox,
                "prediction_bbox": prediction_bbox,
                "crop_shapes": [list(creation_crop.shape), list(prediction_crop.shape)]
            },
            "pixel_analysis": {
                "max_pixel_difference": max_pixel_diff,
                "mean_pixel_difference": mean_pixel_diff,
                "significant_difference": max_pixel_diff > 1.0
            },
            "encoding_from_crops": {
                "distance_from_same_source": crop_encoding_distance,
                "should_be_zero_if_identical": crop_encoding_distance < 1e-6
            },
            "files_saved": {
                "creation_crop": "debug_creation_crop.jpg",
                "prediction_crop": "debug_prediction_crop.jpg",
                "note": "Compare these images visually"
            },
            "conclusion": {
                "problem_is_crops": not crops_identical,
                "problem_is_bboxes": not bboxes_identical,
                "problem_is_elsewhere": crops_identical and crop_encoding_distance > 0.1
            }
        }

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}
