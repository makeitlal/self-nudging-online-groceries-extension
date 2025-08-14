// const overlayedImages = new WeakSet();
const predictionCache = new Map(); // imageURLs

async function classifyImage(imgEl) {
  const imageUrl = imgEl.src;

  // Skip if already processed or invalid
  if (!imageUrl) return;

  // Check cache
  if (predictionCache.has(imageUrl)) {
    const cachedPrediction = predictionCache.get(imageUrl);
    console.log("Using cached prediction for", imageUrl, cachedPrediction);

    if (cachedPrediction.score < 0.5) {
      applyOverlayImage(imgEl);
    }

    // overlayedImages.add(imgEl);
    return;
  }

  const payload = { image_url: imageUrl };
  console.log("Sending payload:", payload);

  try {
    const response = await fetch("https://grocery-extension-group-grocery-store-classifier.hf.space/predict", {
    // const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      console.error("API error:", await response.text());
      return;
    }

    const prediction = await response.json();
    console.log("Prediction result:", prediction);

    // Cache the prediction result
    predictionCache.set(imageUrl, prediction);

    if (prediction.score < 0.5) {
      applyOverlayImage(imgEl);
    }

    // overlayedImages.add(imgEl);
  } catch (err) {
    console.error("Fetch error:", err);
  }
}

function applyOverlayImage(image) {
  image.style.filter = "opacity(0.3)";
  image.style.transition = "filter 0.5s ease";
}

async function scanAllImages() {
  const allImages = document.querySelectorAll("img");
  const promises = Array.from(allImages).map(img => classifyImage(img));
  await Promise.all(promises);
}

scanAllImages();

// Handle dynamic content and scrolling
let scrollTimeout = null;
window.addEventListener("scroll", () => {
  clearTimeout(scrollTimeout);
  scrollTimeout = setTimeout(scanAllImages, 300);
});


// setInterval(scanAllImages, 50000); // Optional: refresh every 50s
