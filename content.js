const unhealthyKeywords = ["chips", "coke", "cola", "soda", "candy", "cookie", "ice cream", "pizza"];

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


function scanAndOverlay() {
  const domain = window.location.hostname;
  const site = Object.keys(parsers).find(key => domain.includes(key));
  if (!site) return;

  const parser = parsers[site];
  const cards = parser.getProductCards();
  
  cards.forEach(card => {
    const text = parser.getText(card) || "";
    const image = parser.getImage(card);
    // if (text && image && isUnhealthy(text)) {
    //   applyOverlayImage(image);
    // }
    if (text && isUnhealthy(text)) {
      applyOverlayCard(card);
    }
  });
}

scanAndOverlay();

// Dynamic updates for scrolling pages
let scrollTimeout = null;
window.addEventListener("scroll", () => {
  clearTimeout(scrollTimeout);
  scrollTimeout = setTimeout(scanAndOverlay, 300);
});

// Periodic rescan (in case of infinite scroll)
setInterval(scanAndOverlay, 5000);
