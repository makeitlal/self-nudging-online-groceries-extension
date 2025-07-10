// content.js
// const unhealthyKeywords = ["chips", "coke", "cola", "soda", "candy", "cookie", "ice cream", "pizza"];
const unhealthyCategories = new Set([
  "WHO NPM Category 1",
  "WHO NPM Category 2",
  "WHO NPM Category 3",
  "WHO NPM Category 4a",
  "WHO NPM Category 4c",
  "WHO NPM Category 5"
]);

const parsers = {
  "instacart.com": {
    getProductCards: () => document.querySelectorAll('div[aria-label="Product"][role="group"]'),
    getText: card => {
      const textEl = card.querySelector('div.e-147kl2c');
      return textEl ? textEl.innerText : "";
    },
    getImage: card => card.querySelector("img"),
  },
  "target.com": {
    getProductCards: () => document.querySelectorAll('[data-test^="item-card-"]'),
    getText: card => {
      const textEl = card.querySelector('[data-test="product-title-sm"]');
      return textEl ? textEl.innerText : "";
    },
    getImage: card => card.querySelector("img"),
  }
};

function isUnhealthy(text) {
  return unhealthyKeywords.some(keyword => text.toLowerCase().includes(keyword));
}
async function classifyProduct(title, imageUrl) {
  const payload = { title: title, image_url: imageUrl };
  console.log("Sending payload:", payload);

  const response = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title, image_url: imageUrl })
  });

  if (!response.ok) {
    console.error("API error:", await response.text());
    return null;
  }

  const data = await response.json();

  return data; // { label, score, [unhealthy] if you implemented it on server }
}

function applyOverlayImage(image) {
  if (image && !image.dataset.overlayApplied) {
    // image.style.filter = "contrast(0.5) grayscale(0.5)";
    image.style.filter = "opacity(0.3)"
    image.style.transition = "filter 0.5s ease";
    // image.style.border = "2px solid red";
    image.dataset.overlayApplied = "true";
  }
}

function applyOverlayCard(card) {
  card.style.opacity = "0.2";
  card.style.transition = "opacity 0.5s ease";
  // card.style.border = "2px solid red";
}


async function scanAndOverlay() {
  const domain = window.location.hostname;
  const site = Object.keys(parsers).find(key => domain.includes(key));
  if (!site) return;

  const parser = parsers[site];
  const cards = parser.getProductCards();

  const promises = Array.from(cards).map(async (card) => {
    const title = parser.getText(card) || "";
    if (!title) return;

    const imgEl = parser.getImage(card);
    const imageUrl = imgEl.src;

    const prediction = await classifyProduct(title, imageUrl);
    if (!prediction) return;

    if (unhealthyCategories.has(prediction.label) && prediction.score > 0.8) {
      applyOverlayCard(card);
    }
  });

  await Promise.all(promises);
}



scanAndOverlay();

// Dynamic updates for scrolling pages
let scrollTimeout = null;
window.addEventListener("scroll", () => {
  clearTimeout(scrollTimeout);
  scrollTimeout = setTimeout(scanAndOverlay, 300);
});

// Periodic rescan (in case of infinite scroll)
setInterval(scanAndOverlay, 50000);
