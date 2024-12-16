import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from torchvision.ops import box_iou


def sliding_window(image, window_size, overlap):
    """
    Gera coordenadas de janelas deslizantes na imagem,
    garantindo que os pixels das bordas também sejam processados.
    """
    step = int(window_size * (1 - overlap))
    h, w, _ = image.shape

    for y in range(0, h, step):
        for x in range(0, w, step):
            x1 = x
            y1 = y
            x2 = min(x + window_size, w)  # Garante que a janela não exceda a largura da imagem
            y2 = min(y + window_size, h)  # Garante que a janela não exceda a altura da imagem
            yield x1, y1, x2, y2


def adjust_coordinates(detections, x_offset, y_offset):
    """
    Ajusta as coordenadas das detecções para as dimensões originais.
    """
    if detections is None or detections.size(0) == 0:
        return detections

    detections = detections.clone()
    detections[:, [0, 2]] += x_offset  # Ajusta x_min e x_max
    detections[:, [1, 3]] += y_offset  # Ajusta y_min e y_max
    return detections


def adjust_keypoints(keypoints, x_offset, y_offset):
    """
    Ajusta as coordenadas dos keypoints para as dimensões originais,
    ignorando valores de 0 (em x e y).
    """
    if keypoints is None or keypoints.size(0) == 0:
        return keypoints

    keypoints = keypoints.clone()

    # Somar o offset apenas onde os valores de keypoints não são 0
    keypoints[:, :, 0] = torch.where(keypoints[:, :, 0] != 0, keypoints[:, :, 0] + x_offset, keypoints[:, :, 0])
    keypoints[:, :, 1] = torch.where(keypoints[:, :, 1] != 0, keypoints[:, :, 1] + y_offset, keypoints[:, :, 1])

    return keypoints


def merge_detections(all_boxes, all_keypoints, iou_threshold=0.8):
    """
    Mescla detecções redundantes usando IOU para caixas delimitadoras.
    """
    if len(all_boxes) == 0:
        return None, None

    # Concatenar as caixas e os keypoints em um tensor
    boxes = torch.cat(all_boxes, dim=0)
    keypoints = torch.cat(all_keypoints, dim=0) if len(all_keypoints) > 0 else torch.empty(0, 0, device=boxes.device)

    # Calcular IOU entre as caixas
    ious = calculate_ious(boxes)

    # Inicializar o vetor de keep como True para todas as caixas
    keep = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)

    # Se houver mais de uma caixa, calcular o keep usando o IOU, considerando o limiar
    if boxes.size(0) > 1:
        for i in range(boxes.size(0)):
            for j in range(i + 1, boxes.size(0)):
                if ious[i, j] > iou_threshold:
                    # Mantenha a caixa com maior confiança (não descartar indiscriminadamente)
                    if boxes[i, 4] < boxes[j, 4]:  # Comparando a confiança
                        keep[i] = False  # Descartar a caixa i se o IOU for maior e a confiança de j for maior
                    else:
                        keep[j] = False  # Descartar a caixa j se o IOU for maior e a confiança de i for maior

    # Garantir que ao menos a primeira caixa seja mantida
    if keep.sum() == 0:
        keep[0] = True

    # Selecionar as caixas e os keypoints que foram mantidos
    kept_boxes = boxes[keep]
    kept_keypoints = keypoints[keep] if keypoints.size(0) > 0 else torch.empty(0, 0, device=boxes.device)

    return kept_boxes, kept_keypoints


def calculate_ious(boxes):
    """
    Calcula o IOU entre as caixas. Aqui, vamos assumir que `boxes` tem a forma (N, 4),
    onde N é o número de caixas e cada caixa é representada por [xmin, ymin, xmax, ymax].
    """
    n = boxes.size(0)
    ious = torch.zeros((n, n), dtype=torch.float32, device=boxes.device)

    for i in range(n):
        for j in range(i + 1, n):
            # Calcular a interseção
            inter_xmin = torch.max(boxes[i, 0], boxes[j, 0])
            inter_ymin = torch.max(boxes[i, 1], boxes[j, 1])
            inter_xmax = torch.min(boxes[i, 2], boxes[j, 2])
            inter_ymax = torch.min(boxes[i, 3], boxes[j, 3])

            inter_area = torch.max(inter_xmax - inter_xmin, torch.tensor(0.0, device=boxes.device)) * \
                         torch.max(inter_ymax - inter_ymin, torch.tensor(0.0, device=boxes.device))

            # Calcular a área de cada caixa
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])

            # Calcular o IOU
            union_area = area_i + area_j - inter_area
            iou = inter_area / union_area if union_area > 0 else torch.tensor(0.0, device=boxes.device)
            ious[i, j] = iou
            ious[j, i] = iou  # IOU é simétrico

    return ious


def process_video(video_path, model_path, window_size, overlap, output_path):
    """
    Processa o vídeo aplicando o modelo YOLOv8-Pose em janelas deslizantes.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    while True:
        ret, frame = cap.read()
        if not ret or frame_number == 800:
            break

        all_boxes = []
        all_keypoints = []

        for x1, y1, x2, y2 in sliding_window(frame, window_size, overlap):
            window = frame[y1:y2, x1:x2]
            results = model.predict(window, device=0, verbose=False, conf=0.3, iou=0.8)

            non_zero_keypoints = (results[0].keypoints.xy != 0).sum(dim=(1, 2))
            if results[0].boxes is not None and len(results[0].boxes) > 0 \
                    and (non_zero_keypoints >= 2).any():
                    # and not torch.all(results[0].keypoints.xy == 0):
                boxes = results[0].boxes.xyxy
                keypoints = results[0].keypoints.xy

                # Adicionando confiança e ID da classe
                confidence = results[0].boxes.conf  # Confiança das detecções
                class_id = results[0].boxes.cls     # ID das classes (no caso, pessoas)

                boxes = adjust_coordinates(boxes, x1, y1)
                keypoints = adjust_keypoints(keypoints, x1, y1)

                # Adicionando confiança e ID da classe
                all_boxes.append(torch.cat((boxes, confidence.unsqueeze(1), class_id.unsqueeze(1)), dim=1))
                all_keypoints.append(keypoints)

        combined_boxes, combined_keypoints = merge_detections(all_boxes, all_keypoints)

        if combined_boxes is not None and combined_boxes.size(0) > 0 and \
            combined_keypoints is not None and not (combined_keypoints == 0).all():
            results = Results(
                orig_img=frame,
                boxes=combined_boxes,
                keypoints=combined_keypoints,
                names={0: "person"},
                path=""
            )
            plotted_frame = results.plot()
        else:
            plotted_frame = frame

        out.write(plotted_frame)
        frame_number += 1
        print(f"Processed frame: {frame_number}", end="\r")

    print(f"Processed frame: {frame_number}")
    cap.release()
    out.release()
    print("\nProcessing complete.")


def process_video_batch(video_path, model_path, window_size, overlap, output_path, batch_size=16):
    """
    Processa o vídeo aplicando o modelo YOLOv8-Pose em janelas deslizantes usando batches.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

    # Armazenar múltiplos frames para processamento em batch
    frames_batch = []
    frame_indices = []

    while True:
        ret, frame = cap.read()
        if not ret or frame_number == 800:
            break

        frames_batch.append(frame)
        frame_indices.append(frame_number)
        frame_number += 1

        # Processar quando atingimos o batch_size ou quando o vídeo termina
        if len(frames_batch) == batch_size or not ret:
            all_detections = []  # Para armazenar detecções de todos os frames no batch

            for frame_idx, frame in enumerate(frames_batch):
                all_boxes = []
                all_keypoints = []

                for x1, y1, x2, y2 in sliding_window(frame, window_size, overlap):
                    window = frame[y1:y2, x1:x2]
                    results = model.predict(window, device=0, verbose=False, conf=0.3, iou=0.8)

                    non_zero_keypoints = (results[0].keypoints.xy != 0).sum(dim=(1, 2))
                    if results[0].boxes is not None and len(results[0].boxes) > 0 \
                            and (non_zero_keypoints >= 2).any():
                        boxes = results[0].boxes.xyxy
                        keypoints = results[0].keypoints.xy

                        # Adicionando confiança e ID da classe
                        confidence = results[0].boxes.conf  # Confiança das detecções
                        class_id = results[0].boxes.cls     # ID das classes (no caso, pessoas)

                        boxes = adjust_coordinates(boxes, x1, y1)
                        keypoints = adjust_keypoints(keypoints, x1, y1)

                        # Adicionando confiança e ID da classe
                        all_boxes.append(torch.cat((boxes, confidence.unsqueeze(1), class_id.unsqueeze(1)), dim=1))
                        all_keypoints.append(keypoints)

                combined_boxes, combined_keypoints = merge_detections(all_boxes, all_keypoints)
                all_detections.append((combined_boxes, combined_keypoints))

            # Após coletar as detecções do batch, renderizar os frames
            for frame_idx, (combined_boxes, combined_keypoints) in enumerate(all_detections):
                frame = frames_batch[frame_idx]

                if combined_boxes is not None and combined_boxes.size(0) > 0 and \
                    combined_keypoints is not None and not (combined_keypoints == 0).all():
                    results = Results(
                        orig_img=frame,
                        boxes=combined_boxes,
                        keypoints=combined_keypoints,
                        names={0: "person"},
                        path=""
                    )
                    plotted_frame = results.plot()
                else:
                    plotted_frame = frame

                out.write(plotted_frame)
                print(f"Processed frame: {frame_indices[frame_idx]}", end="\r")

            # Limpar o batch
            frames_batch = []
            frame_indices = []

    print("\nProcessing complete.")
    cap.release()
    out.release()


# Configurações
video_path = "/path/video/name.mp4"
model_path = "yolov8n-pose.pt"  # Substitua pelo caminho do modelo treinado
output_path = "processed_video.mp4"
window_size = 640
overlap = 0.2

# process_video(video_path, model_path, window_size, overlap, output_path)
process_video_batch(video_path, model_path, window_size, overlap, output_path)

