"""
analise_histopatologia.py

Versão acadêmica e reprodutível do pipeline de análise morfológica
exploratória de imagens histopatológicas.

DIFERENÇAS DESTA VERSÃO
-----------------------
- exporta métricas em CSV
- exporta métricas em JSON
- gera figura-resumo do pipeline
- possui comentários detalhados
- adequado para repositório de estudo / IC / TCC

AVISO
-----
Este script NÃO diagnostica câncer.
Ele NÃO substitui laudo anatomopatológico.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology, segmentation


@dataclass
class NuclearMetrics:
    """Estrutura para armazenar métricas agregadas dos objetos segmentados."""
    nuclei_count: int
    nuclei_density: float
    mean_area: float
    std_area: float
    cv_area: float
    mean_perimeter: float
    mean_circularity: float
    mean_intensity: float
    std_intensity: float
    atypia_score: float


def load_image(image_path: str) -> np.ndarray:
    """
    Carrega a imagem do disco.

    Escolha:
    - usamos OpenCV pela robustez e simplicidade;
    - convertemos BGR -> RGB para manter consistência visual.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normaliza cada canal de cor separadamente para [0, 255].

    Motivo:
    - reduzir variações grosseiras de brilho/contraste;
    - deixar o pipeline um pouco mais estável entre imagens.
    """
    img_float = img.astype(np.float32)
    norm = np.zeros_like(img_float)

    for c in range(3):
        norm[:, :, c] = cv2.normalize(img_float[:, :, c], None, 0, 255, cv2.NORM_MINMAX)

    return norm.astype(np.uint8)


def extract_nuclear_channel(img_rgb: np.ndarray) -> np.ndarray:
    """
    Cria uma máscara aproximada para regiões nucleares.

    Estratégia:
    - HSV: procura tons mais arroxeados/escuros;
    - Lab: adiciona apoio cromático;
    - percentis deixam a regra um pouco mais adaptável.

    Limitação:
    - é um método heurístico;
    - não substitui segmentação supervisionada.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    purple_mask = (((h > 120) & (h < 175)) & (s > 20) & (v < 200))
    dark_mask = v < np.percentile(v, 55)
    lab_mask = a > np.percentile(a, 45)

    combined = (purple_mask | (dark_mask & lab_mask)).astype(np.uint8) * 255
    return combined


def segment_nuclei(mask: np.ndarray, min_area: int = 40, max_area: int = 15000) -> np.ndarray:
    """
    Limpa e rotula objetos da máscara.

    Etapas:
    - abertura: remove ruído pontual;
    - fechamento: fecha pequenas falhas;
    - remove_small_objects: limpa resíduos;
    - remove_small_holes: preenche pequenos buracos;
    - filtra por área.
    """
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    cleaned_bool = cleaned > 0
    cleaned_bool = morphology.remove_small_objects(cleaned_bool, min_size=min_area)
    cleaned_bool = morphology.remove_small_holes(cleaned_bool, area_threshold=80)

    labels = measure.label(cleaned_bool)
    props = measure.regionprops(labels)

    filtered = np.zeros_like(labels, dtype=np.uint16)
    current_label = 1

    for prop in props:
        if min_area <= prop.area <= max_area:
            filtered[labels == prop.label] = current_label
            current_label += 1

    return filtered


def compute_metrics(img_rgb: np.ndarray, labels: np.ndarray) -> NuclearMetrics:
    """
    Extrai métricas morfológicas e de intensidade.

    Métricas escolhidas por serem intuitivas:
    - área média;
    - variação de área (anisocariose);
    - circularidade;
    - intensidade;
    - densidade nuclear.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    props = measure.regionprops(labels, intensity_image=gray)

    if len(props) == 0:
        return NuclearMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    areas = []
    perimeters = []
    circularities = []
    intensities = []

    for prop in props:
        area = float(prop.area)
        perimeter = float(prop.perimeter) if prop.perimeter > 0 else 1.0
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)

        areas.append(area)
        perimeters.append(perimeter)
        circularities.append(circularity)
        intensities.append(float(prop.mean_intensity))

    areas = np.array(areas, dtype=np.float32)
    perimeters = np.array(perimeters, dtype=np.float32)
    circularities = np.array(circularities, dtype=np.float32)
    intensities = np.array(intensities, dtype=np.float32)

    img_area = img_rgb.shape[0] * img_rgb.shape[1]
    nuclei_density = len(props) / img_area

    mean_area = float(np.mean(areas))
    std_area = float(np.std(areas))
    cv_area = float(std_area / mean_area) if mean_area > 0 else 0.0
    mean_perimeter = float(np.mean(perimeters))
    mean_circularity = float(np.mean(circularities))
    mean_intensity = float(np.mean(intensities))
    std_intensity = float(np.std(intensities))

    # Escore exploratório, não validado.
    area_component = min(cv_area / 0.60, 1.0)
    shape_component = min((1.0 - mean_circularity) / 0.50, 1.0)
    dark_component = min((150.0 - mean_intensity) / 80.0, 1.0)
    density_component = min(nuclei_density * 20000.0, 1.0)
    components = [max(0.0, x) for x in [area_component, shape_component, dark_component, density_component]]
    atypia_score = float(np.mean(components) * 100.0)

    return NuclearMetrics(
        nuclei_count=len(props),
        nuclei_density=nuclei_density,
        mean_area=mean_area,
        std_area=std_area,
        cv_area=cv_area,
        mean_perimeter=mean_perimeter,
        mean_circularity=mean_circularity,
        mean_intensity=mean_intensity,
        std_intensity=std_intensity,
        atypia_score=atypia_score,
    )


def overlay_nuclei(img_rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Desenha os contornos dos objetos segmentados sobre a imagem."""
    overlay = img_rgb.copy()
    boundaries = segmentation.find_boundaries(labels > 0, mode="outer")
    overlay[boundaries] = [255, 0, 0]
    return overlay


def create_label_visualization(labels: np.ndarray) -> np.ndarray:
    """
    Gera uma visualização colorida dos objetos segmentados.

    Útil para inspeção humana.
    """
    label_vis = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for lb in np.unique(labels):
        if lb == 0:
            continue
        color = rng.integers(50, 255, size=3, dtype=np.uint8)
        label_vis[labels == lb] = color
    return label_vis


def generate_report(metrics: NuclearMetrics) -> str:
    """Traduz as métricas em um laudo descritivo automatizado."""
    if metrics.nuclei_count == 0:
        return (
            "Laudo automatizado descritivo:\n"
            "- Não foi possível segmentar núcleos com os parâmetros atuais.\n"
            "- Recomenda-se revisar a qualidade da imagem, coloração e limiares.\n"
            "- Sem condições de inferência morfológica automatizada.\n"
        )

    achados = []

    if metrics.cv_area > 0.45:
        achados.append("variação importante do tamanho nuclear (anisocariose)")
    elif metrics.cv_area > 0.30:
        achados.append("variação moderada do tamanho nuclear")

    if metrics.mean_circularity < 0.65:
        achados.append("irregularidade moderada a acentuada do contorno nuclear")
    elif metrics.mean_circularity < 0.80:
        achados.append("leve irregularidade do contorno nuclear")

    if metrics.mean_intensity < 120:
        achados.append("núcleos globalmente hipercromáticos")
    elif metrics.mean_intensity < 145:
        achados.append("tendência a hipercromasia nuclear")

    if metrics.nuclei_density > 0.00008:
        achados.append("aumento da densidade nuclear")
    elif metrics.nuclei_density > 0.00004:
        achados.append("densidade nuclear discretamente aumentada")

    if not achados:
        achados.append("sem alterações morfológicas automáticas exuberantes pelos critérios heurísticos adotados")

    if metrics.atypia_score >= 70:
        impressao = (
            "Achados automatizados sugestivos de atipia celular relevante. "
            "A imagem merece correlação com avaliação histopatológica formal."
        )
    elif metrics.atypia_score >= 45:
        impressao = (
            "Achados automatizados compatíveis com atipia discreta a moderada, sem caráter conclusivo."
        )
    else:
        impressao = (
            "Análise automatizada sem forte evidência morfológica de atipia importante pelos critérios adotados."
        )

    disclaimer = (
        "Este relatório é exploratório e não substitui diagnóstico anatomopatológico. "
        "Não permite confirmar ou excluir câncer sem contexto arquitetural, lâmina completa, "
        "correlação clínico-patológica e avaliação por patologista."
    )

    report = [
        "Laudo automatizado descritivo",
        "",
        "Métricas extraídas:",
        f"- Número de núcleos segmentados: {metrics.nuclei_count}",
        f"- Densidade nuclear: {metrics.nuclei_density:.8f}",
        f"- Área nuclear média: {metrics.mean_area:.2f} px²",
        f"- Desvio-padrão da área nuclear: {metrics.std_area:.2f}",
        f"- Coeficiente de variação da área: {metrics.cv_area:.2f}",
        f"- Perímetro nuclear médio: {metrics.mean_perimeter:.2f}",
        f"- Circularidade média: {metrics.mean_circularity:.2f}",
        f"- Intensidade média dos núcleos (escala de cinza): {metrics.mean_intensity:.2f}",
        f"- Desvio-padrão da intensidade: {metrics.std_intensity:.2f}",
        f"- Escore heurístico de atipia: {metrics.atypia_score:.1f}/100",
        "",
        "Achados descritivos sugeridos:",
    ]

    for a in achados:
        report.append(f"- {a}")

    report.extend([
        "",
        "Impressão:",
        impressao,
        "",
        "Observação importante:",
        disclaimer,
    ])

    return "\n".join(report)


def save_metrics_csv(metrics: NuclearMetrics, output_path: str) -> None:
    """Exporta as métricas em CSV para fácil abertura em Excel."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics).keys()))
        writer.writeheader()
        writer.writerow(asdict(metrics))


def save_metrics_json(metrics: NuclearMetrics, output_path: str) -> None:
    """Exporta as métricas em JSON para integração com outros scripts."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, ensure_ascii=False, indent=2)


def save_pipeline_figure(original: np.ndarray, mask: np.ndarray, label_vis: np.ndarray, overlay: np.ndarray, output_path: str) -> None:
    """
    Gera uma figura-resumo com as principais etapas do pipeline.

    Escolha:
    - matplotlib ajuda a produzir uma imagem de documentação científica.
    """
    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(original)
    ax1.set_title("Imagem original / normalizada")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(mask, cmap="gray")
    ax2.set_title("Máscara nuclear")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(label_vis)
    ax3.set_title("Segmentação colorida")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(overlay)
    ax4.set_title("Overlay de contornos")
    ax4.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_outputs(output_dir: str, original: np.ndarray, mask: np.ndarray, labels: np.ndarray, overlay: np.ndarray, report: str, metrics: NuclearMetrics) -> None:
    """
    Salva todas as saídas do pipeline em disco.
    """
    os.makedirs(output_dir, exist_ok=True)

    label_vis = create_label_visualization(labels)

    cv2.imwrite(os.path.join(output_dir, "01_original.png"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "02_mask_nuclear.png"), mask)
    cv2.imwrite(os.path.join(output_dir, "03_segmentacao_colorida.png"), cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "04_overlay_contornos.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    with open(os.path.join(output_dir, "laudo_automatizado.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    save_metrics_csv(metrics, os.path.join(output_dir, "metricas.csv"))
    save_metrics_json(metrics, os.path.join(output_dir, "metricas.json"))
    save_pipeline_figure(original, mask, label_vis, overlay, os.path.join(output_dir, "05_resumo_pipeline.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise morfológica exploratória de imagem histopatológica.")
    parser.add_argument("--image", required=True, help="Caminho da imagem.")
    parser.add_argument("--output", default="saida_analise", help="Pasta de saída.")
    args = parser.parse_args()

    img = load_image(args.image)
    img_norm = normalize_image(img)
    mask = extract_nuclear_channel(img_norm)
    labels = segment_nuclei(mask)
    metrics = compute_metrics(img_norm, labels)
    overlay = overlay_nuclei(img_norm, labels)
    report = generate_report(metrics)

    save_outputs(args.output, img_norm, mask, labels, overlay, report, metrics)

    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)
    print(f"\nArquivos salvos em: {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()
