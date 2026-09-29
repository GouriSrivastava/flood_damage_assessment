import cv2
import numpy as np

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit(
        "Gradio is not installed. Install the app dependencies from "
        "requirements-flood-app.txt."
    ) from exc

if not hasattr(gr, "Blocks"):
    raise SystemExit(
        "The Gradio installation is incomplete or incompatible. "
        "Reinstall the app dependencies from requirements-flood-app.txt."
    )


def create_flood_demo():
    h, w = 512, 768

    pre = np.zeros((h, w, 3), dtype=np.uint8)
    pre[:] = [120, 140, 100]

    cv2.rectangle(pre, (0, 0), (768, 120), (140, 150, 110), -1)
    cv2.rectangle(pre, (0, 390), (768, 512), (130, 145, 105), -1)
    cv2.line(pre, (0, 350), (768, 120), (170, 160, 135), 20)
    cv2.line(pre, (150, 0), (300, 512), (170, 160, 135), 15)

    buildings = [
        (320, 50, 390, 110),
        (430, 60, 510, 125),
        (550, 40, 640, 110),
        (280, 180, 365, 250),
        (400, 185, 480, 255),
        (520, 165, 610, 240),
        (330, 310, 420, 370),
        (470, 300, 560, 365),
        (590, 285, 675, 350),
    ]

    for x1, y1, x2, y2 in buildings:
        cv2.rectangle(pre, (x1, y1), (x2, y2), (190, 185, 170), -1)
        cv2.rectangle(pre, (x1 + 8, y1 + 8), (x2 - 8, y2 - 8), (210, 205, 190), -1)

    post = pre.copy()
    flood_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.ellipse(flood_mask, (230, 330), (220, 130), 0, 0, 360, 255, -1)
    cv2.rectangle(flood_mask, (0, 390), (768, 512), 255, -1)

    post[flood_mask > 0] = [70, 105, 145]

    pts = np.array([
        [120, 250],
        [270, 160],
        [380, 190],
        [340, 300],
        [200, 340],
    ])

    cv2.fillPoly(flood_mask, [pts], 255)
    post[flood_mask > 0] = [65, 100, 140]

    damaged_buildings = [
        (280, 180, 365, 250),
        (520, 165, 610, 240),
        (470, 300, 560, 365),
    ]

    for x1, y1, x2, y2 in damaged_buildings:
        cv2.rectangle(post, (x1, y1), (x2, y2), (80, 75, 70), -1)

    return pre, post, flood_mask


def _prepare_image(img):
    if img is None:
        raise gr.Error("Upload both the before-flood and after-flood images.")

    img = np.asarray(img)

    if img.size == 0:
        raise gr.Error("The uploaded image is empty. Choose a valid image file.")

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim != 3 or img.shape[-1] != 3:
        raise gr.Error("Images must be grayscale, RGB, or RGBA.")

    if not np.issubdtype(img.dtype, np.number) or not np.isfinite(img).all():
        raise gr.Error("The uploaded image contains invalid pixel values.")

    if np.issubdtype(img.dtype, np.floating) and img.size and img.max() <= 1.0 and img.min() >= 0.0:
        img = img * 255.0

    img = np.clip(img, 0, 255).astype(np.uint8)

    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def flood_damage_pipeline(pre, post, rainfall_72h=120, peak_intensity=18, humidity=82, population_density_value=1200):
    pre = _prepare_image(pre)
    post = _prepare_image(post)

    if pre.shape[:2] != post.shape[:2]:
        pre_ratio = pre.shape[1] / pre.shape[0]
        post_ratio = post.shape[1] / post.shape[0]
        if abs(pre_ratio - post_ratio) / pre_ratio > 0.02:
            raise gr.Error(
                "The before and after images have different aspect ratios. "
                "Use images covering the same area with matching proportions."
            )

        interpolation = (
            cv2.INTER_AREA
            if post.shape[0] > pre.shape[0] or post.shape[1] > pre.shape[1]
            else cv2.INTER_LINEAR
        )
        post = cv2.resize(
            post,
            (pre.shape[1], pre.shape[0]),
            interpolation=interpolation,
        )

    gray_pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(gray_pre, gray_post)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)

    threshold_value = np.percentile(difference, 88)
    change_mask = (difference > threshold_value).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
    change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)

    changed = change_mask > 0

    R = post[:, :, 0].astype(np.int16)
    G = post[:, :, 1].astype(np.int16)
    B = post[:, :, 2].astype(np.int16)

    water_mask = ((B - R > 15) & (B > G)) & changed

    rain_score = min(rainfall_72h / 150, 1.0)
    peak_score = min(peak_intensity / 30, 1.0)
    humidity_score = humidity / 100.0
    weather_score = 0.35 * rain_score + 0.35 * peak_score + 0.30 * humidity_score

    change_score = difference.astype(np.float32) / 255.0
    severity_score = 0.60 * change_score + 0.25 * water_mask.astype(float) + 0.15 * weather_score
    severity_score = np.clip(severity_score, 0, 1)

    severe = (severity_score >= 0.65) & changed
    moderate = (severity_score >= 0.40) & (severity_score < 0.65) & changed
    peripheral = (severity_score < 0.40) & changed

    if np.any(changed):
        visual_strength = np.mean(difference[changed]) / 255.0
    else:
        visual_strength = 0.0

    visual_strength = np.clip(visual_strength, 0, 1)
    confidence = 0.70 * visual_strength + 0.30 * weather_score
    confidence = np.clip(confidence, 0, 1)

    h, w = post.shape[:2]
    affected = severe | moderate | peripheral
    affected_fraction = np.sum(affected) / (h * w)
    scene_area_km2 = 4.0
    affected_area = affected_fraction * scene_area_km2
    affected_population = affected_area * population_density_value

    if confidence < 0.60:
        review = "⚠️ MANUAL REVIEW REQUIRED"
    else:
        review = "✅ HIGH CONFIDENCE"

    overlay = post.copy()
    overlay[peripheral] = [255, 200, 0]
    overlay[moderate] = [255, 100, 0]
    overlay[severe] = [220, 0, 0]
    damage_map = cv2.addWeighted(post, 0.55, overlay, 0.45, 0)

    flood_percentage = (np.sum(water_mask) / (h * w)) * 100
    severe_percentage = (np.sum(severe) / (h * w)) * 100
    moderate_percentage = (np.sum(moderate) / (h * w)) * 100
    peripheral_percentage = (np.sum(peripheral) / (h * w)) * 100

    result_dict = {
        "Flood/Wet Area (%)": round(float(flood_percentage), 2),
        "Severe Damage (%)": round(float(severe_percentage), 2),
        "Moderate Damage (%)": round(float(moderate_percentage), 2),
        "Peripheral Risk (%)": round(float(peripheral_percentage), 2),
        "Affected Area (km²)": round(float(affected_area), 2),
        "Estimated Affected Population": round(float(affected_population), 0),
        "Confidence (%)": round(float(confidence * 100), 1),
        "Weather Corroboration (%)": round(float(weather_score * 100), 1),
        "Review Status": review,
    }

    summary = (
        "### Flood Damage Assessment Summary\n"
        f"- Flood/Wet Area: {result_dict['Flood/Wet Area (%)']:.2f}%\n"
        f"- Severe Damage: {result_dict['Severe Damage (%)']:.2f}%\n"
        f"- Moderate Damage: {result_dict['Moderate Damage (%)']:.2f}%\n"
        f"- Peripheral Risk: {result_dict['Peripheral Risk (%)']:.2f}%\n"
        f"- Affected Area: {result_dict['Affected Area (km²)']:.2f} km²\n"
        f"- Estimated Population: {result_dict['Estimated Affected Population']:.0f}\n"
        f"- Confidence: {result_dict['Confidence (%)']:.1f}%\n"
        f"- Status: {review}"
    )

    return (
        cv2.cvtColor(damage_map, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(change_mask, cv2.COLOR_GRAY2RGB),
        (water_mask.astype(np.uint8) * 255),
        result_dict,
        summary,
    )


def analyze_images(pre_img, post_img, rainfall_72h=120, peak_intensity=18, humidity=82, population_density=1200):
    damage_map, change_mask, water_mask, result_dict, summary = flood_damage_pipeline(
        pre_img,
        post_img,
        rainfall_72h=rainfall_72h,
        peak_intensity=peak_intensity,
        humidity=humidity,
        population_density_value=population_density,
    )
    return damage_map, change_mask, water_mask, result_dict, summary


pre_demo, post_demo, _ = create_flood_demo()
pre_demo = cv2.cvtColor(pre_demo, cv2.COLOR_BGR2RGB)
post_demo = cv2.cvtColor(post_demo, cv2.COLOR_BGR2RGB)

with gr.Blocks(title="Flood Damage Assessment") as demo:
    gr.Markdown("# Flood Damage Assessment")
    gr.Markdown("Upload before and after flood images, or use the demo scene to test the pipeline.")

    with gr.Row():
        pre_input = gr.Image(label="Before Flood (Pre-image)", type="numpy", value=pre_demo)
        post_input = gr.Image(label="After Flood (Post-image)", type="numpy", value=post_demo)

    with gr.Row():
        rainfall = gr.Slider(minimum=0, maximum=300, value=120, step=1, label="72h Rainfall (mm)")
        peak = gr.Slider(minimum=0, maximum=60, value=18, step=1, label="Peak Intensity (mm/h)")
        humidity = gr.Slider(minimum=0, maximum=100, value=82, step=1, label="Humidity (%)")
        population_density = gr.Slider(minimum=0, maximum=5000, value=1200, step=10, label="Population Density (people/km²)")

    submit_btn = gr.Button("Analyze Flood Damage")

    with gr.Row():
        damage_output = gr.Image(label="Damage Severity Map")
        change_output = gr.Image(label="Change Detection Mask")
        water_output = gr.Image(label="Water Detection Mask")

    result_json = gr.JSON(label="Assessment Summary")
    result_text = gr.Markdown(label="Summary")

    submit_btn.click(
        fn=analyze_images,
        inputs=[pre_input, post_input, rainfall, peak, humidity, population_density],
        outputs=[damage_output, change_output, water_output, result_json, result_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
