from pathlib import Path
import shutil
import tempfile
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"
RUNTIME_MODEL_DIR = Path(tempfile.gettempdir()) / "lraise_runtime"
RUNTIME_MODEL_PATH = RUNTIME_MODEL_DIR / "pose_landmarker_lite.task"
CAMERA_WINDOW_TITLE = "Camera | Elevacao Lateral"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)

POSE_CONNECTIONS = [
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
]

POSE_POINTS = [11, 12, 13, 14, 15, 16]
MAX_HISTORY = 30
SHOULDER_BASELINE_FRAMES = 30
DOWN_THRESHOLD_OFFSET = 0.16
TOP_POSITION_OFFSET = 0.03
WRIST_ELBOW_MARGIN = 0.02
SHOULDER_SHRUG_MARGIN = 0.03
REP_SCORE_INVALID_THRESHOLD = 70
MIN_REP_FRAMES = 6
UP_TARGET_CHEST_OFFSET = 0.10
MAX_MISSED_POSE_FRAMES = 8
ROTATE_CAMERA_180 = False
POSE_NOT_DETECTED_LABEL = "Pose nao detectada"

COLOR_SUCCESS = (91, 170, 120)
COLOR_WARNING = (80, 155, 230)
COLOR_ALERT = (85, 90, 220)
COLOR_PATH_OK = (90, 220, 250)
COLOR_PATH_ALERT = (85, 90, 220)


def ensure_model():
    if not MODEL_PATH.exists():
        print("Baixando modelo de pose...")
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print("Modelo baixado com sucesso.")


def prepare_runtime_model():
    # O MediaPipe pode falhar em caminhos com caracteres especiais no Windows.
    RUNTIME_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    should_copy = not RUNTIME_MODEL_PATH.exists()
    if not should_copy:
        should_copy = MODEL_PATH.stat().st_size != RUNTIME_MODEL_PATH.stat().st_size

    if should_copy:
        shutil.copyfile(MODEL_PATH, RUNTIME_MODEL_PATH)

    return RUNTIME_MODEL_PATH


def create_detector():
    runtime_model_path = prepare_runtime_model()
    base_options = python.BaseOptions(model_asset_path=str(runtime_model_path))

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.PoseLandmarker.create_from_options(options)


def get_phase_label(movement_phase):
    return "subindo" if movement_phase == "up" else "descendo"


def draw_upper_body(frame, landmarks):
    h, w = frame.shape[:2]

    for start_idx, end_idx in POSE_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * w)
        y1 = int(start.y * h)
        x2 = int(end.x * w)
        y2 = int(end.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    for idx in POSE_POINTS:
        landmark = landmarks[idx]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)


def draw_path(frame, points, color):
    if len(points) < 2:
        return

    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, 2)


def update_baseline(landmarks, shoulder_samples, locked):
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2

    if not locked and len(shoulder_samples) < SHOULDER_BASELINE_FRAMES:
        shoulder_samples.append(shoulder_y)

    if not shoulder_samples:
        return None

    return sum(shoulder_samples) / len(shoulder_samples)


def update_rep_state(landmarks, movement_phase):
    shoulder_y = landmarks[12].y
    wrist_y = landmarks[16].y

    up_threshold = shoulder_y + UP_TARGET_CHEST_OFFSET
    down_threshold = shoulder_y + DOWN_THRESHOLD_OFFSET
    rep_started_now = False
    rep_completed_now = False

    if movement_phase == "down" and wrist_y < up_threshold:
        movement_phase = "up"
        rep_started_now = True
    elif movement_phase == "up" and wrist_y > down_threshold:
        movement_phase = "down"
        rep_completed_now = True

    return movement_phase, rep_started_now, rep_completed_now


def validate_wrist_above_elbow(landmarks):
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_top = left_wrist.y < (left_shoulder.y + TOP_POSITION_OFFSET)
    right_top = right_wrist.y < (right_shoulder.y + TOP_POSITION_OFFSET)

    left_error = left_top and left_wrist.y < (left_elbow.y - WRIST_ELBOW_MARGIN)
    right_error = right_top and right_wrist.y < (right_elbow.y - WRIST_ELBOW_MARGIN)

    return left_error or right_error


def validate_shrug(landmarks, shoulder_baseline):
    if shoulder_baseline is None:
        return False

    current_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    return current_shoulder_y < (shoulder_baseline - SHOULDER_SHRUG_MARGIN)


def get_feedback(landmarks, shoulder_baseline):
    errors = []

    if validate_wrist_above_elbow(landmarks):
        errors.append("Pulso acima do cotovelo")

    if validate_shrug(landmarks, shoulder_baseline):
        errors.append("Ombros encolhidos")

    return errors


def get_feedback_state(is_tracking, movement_phase, errors, shoulder_baseline_ready):
    if errors and errors[0] == POSE_NOT_DETECTED_LABEL:
        return (
            "Ajuste seu enquadramento",
            "Fique de frente para a camera e mantenha os bracos visiveis.",
            COLOR_WARNING,
        )

    if not shoulder_baseline_ready:
        return (
            "Calibrando postura inicial",
            "Fique parado de frente para a camera por alguns segundos.",
            COLOR_WARNING,
        )

    if not is_tracking:
        return (
            "Aquecimento em andamento",
            "Faca uma repeticao completa para iniciar a analise.",
            COLOR_WARNING,
        )

    if errors:
        return ("Ajuste sua execucao", errors[0], COLOR_ALERT)

    return (
        "Movimento sendo analisado",
        f"Fase atual: {get_phase_label(movement_phase)}.",
        COLOR_SUCCESS,
    )


def draw_feedback(
    frame,
    is_tracking,
    movement_phase,
    errors,
    shoulder_baseline_ready,
    rep_count,
    valid_rep_count,
):
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (frame.shape[1] - 14, 88), (26, 34, 47), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    main_text, detail_text, color = get_feedback_state(
        is_tracking=is_tracking,
        movement_phase=movement_phase,
        errors=errors,
        shoulder_baseline_ready=shoulder_baseline_ready,
    )

    cv2.putText(
        frame,
        main_text,
        (28, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        detail_text,
        (28, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (218, 224, 232),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Repeticoes: {rep_count} | validas: {valid_rep_count} | Q ou ESC para sair",
        (28, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 239, 244),
        1,
        cv2.LINE_AA,
    )


def finalize_rep(rep_index, rep_frame_count, rep_errors):
    error_frames = sum(1 for errors in rep_errors if errors)
    score = 0

    if rep_frame_count:
        score = max(0, round(100 * (1 - error_frames / rep_frame_count)))

    error_kinds = sorted({error for errors in rep_errors for error in errors})
    is_valid = rep_frame_count > 0 and score >= REP_SCORE_INVALID_THRESHOLD

    return {
        "index": rep_index,
        "valid": is_valid,
        "errors": error_kinds,
    }


def should_count_rep(rep_frame_count):
    return rep_frame_count >= MIN_REP_FRAMES


def reset_active_rep():
    return False, 0, []


def print_rep_result(rep_summary):
    if rep_summary["valid"]:
        print(f"Repeticao {rep_summary['index']}: correta")
        return

    causes = ", ".join(rep_summary["errors"]) if rep_summary["errors"] else "sem causa identificada"
    print(f"Repeticao {rep_summary['index']}: incorreta ({causes})")


def main():
    ensure_model()
    detector = create_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nao foi possivel acessar a camera.")
        detector.close()
        return

    frame_count = 0
    wrist_history = []
    shoulder_samples = []
    movement_phase = "down"
    first_rep_completed = False
    is_tracking = False
    rep_active = False
    current_rep_frame_count = 0
    current_rep_errors = []
    total_rep_count = 0
    valid_rep_count = 0
    missed_pose_frames = 0

    print("Camera iniciada.")
    print("A primeira repeticao completa serve como aquecimento para iniciar a analise.")
    print("Resultados das repeticoes:")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Nao foi possivel ler a camera.")
                break

            if ROTATE_CAMERA_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_count += 1
            timestamp_ms = int(frame_count * 1000 / 30)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            errors = []

            if result.pose_landmarks:
                missed_pose_frames = 0
                h, w = frame.shape[:2]
                landmarks = result.pose_landmarks[0]
                draw_upper_body(frame, landmarks)

                shoulder_baseline = update_baseline(
                    landmarks,
                    shoulder_samples,
                    locked=first_rep_completed,
                )
                movement_phase, rep_started_now, rep_completed_now = update_rep_state(
                    landmarks,
                    movement_phase,
                )

                right_wrist = landmarks[16]
                wrist_x = int(right_wrist.x * w)
                wrist_y = int(right_wrist.y * h)

                if rep_completed_now and not first_rep_completed:
                    first_rep_completed = True
                    is_tracking = True
                    rep_active, current_rep_frame_count, current_rep_errors = reset_active_rep()
                    wrist_history.clear()

                errors = get_feedback(landmarks, shoulder_baseline) if is_tracking else []
                path_color = COLOR_PATH_ALERT if errors else COLOR_PATH_OK

                if is_tracking and rep_started_now and not rep_active:
                    rep_active = True
                    current_rep_frame_count = 0
                    current_rep_errors = []

                if is_tracking:
                    wrist_history.append((wrist_x, wrist_y))
                    if len(wrist_history) > MAX_HISTORY:
                        wrist_history.pop(0)
                    draw_path(frame, wrist_history, path_color)

                if rep_active:
                    current_rep_frame_count += 1
                    current_rep_errors.append(errors[:])

                if is_tracking and rep_completed_now and rep_active:
                    if should_count_rep(current_rep_frame_count):
                        total_rep_count += 1
                        rep_summary = finalize_rep(
                            total_rep_count,
                            current_rep_frame_count,
                            current_rep_errors,
                        )
                        if rep_summary["valid"]:
                            valid_rep_count += 1
                        print_rep_result(rep_summary)

                    rep_active, current_rep_frame_count, current_rep_errors = reset_active_rep()

                draw_feedback(
                    frame,
                    is_tracking=is_tracking,
                    movement_phase=movement_phase,
                    errors=errors,
                    shoulder_baseline_ready=len(shoulder_samples) >= SHOULDER_BASELINE_FRAMES,
                    rep_count=total_rep_count,
                    valid_rep_count=valid_rep_count,
                )
            else:
                missed_pose_frames += 1
                if missed_pose_frames >= MAX_MISSED_POSE_FRAMES:
                    rep_active, current_rep_frame_count, current_rep_errors = reset_active_rep()
                    movement_phase = "down"

                draw_feedback(
                    frame,
                    is_tracking=is_tracking,
                    movement_phase=movement_phase,
                    errors=[POSE_NOT_DETECTED_LABEL],
                    shoulder_baseline_ready=len(shoulder_samples) >= SHOULDER_BASELINE_FRAMES,
                    rep_count=total_rep_count,
                    valid_rep_count=valid_rep_count,
                )

            cv2.imshow(CAMERA_WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
