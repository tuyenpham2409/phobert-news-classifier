// Category styling configurations
const CATEGORY_CONFIGS = {
    'THỂ THAO': {
        bg: 'bg-red-50',
        text: 'text-red-700',
        border: 'border-red-200',
        icon: 'fas fa-futbol',
        iconColor: 'text-red-500',
        progressColor: '#EF4444'
    },
    'SỨC KHỎE': {
        bg: 'bg-emerald-50',
        text: 'text-emerald-700',
        border: 'border-emerald-200',
        icon: 'fas fa-heartbeat',
        iconColor: 'text-emerald-500',
        progressColor: '#10B981'
    },
    'GIÁO DỤC': {
        bg: 'bg-blue-50',
        text: 'text-blue-700',
        border: 'border-blue-200',
        icon: 'fas fa-graduation-cap',
        iconColor: 'text-blue-500',
        progressColor: '#3B82F6'
    },
    'PHÁP LUẬT': {
        bg: 'bg-indigo-50',
        text: 'text-indigo-700',
        border: 'border-indigo-200',
        icon: 'fas fa-gavel',
        iconColor: 'text-indigo-500',
        progressColor: '#6366F1'
    },
    'KINH DOANH': {
        bg: 'bg-amber-50',
        text: 'text-amber-700',
        border: 'border-amber-200',
        icon: 'fas fa-briefcase',
        iconColor: 'text-amber-500',
        progressColor: '#F59E0B'
    },
    'THƯ GIÃN': {
        bg: 'bg-purple-50',
        text: 'text-purple-700',
        border: 'border-purple-200',
        icon: 'fas fa-smile',
        iconColor: 'text-purple-500',
        progressColor: '#8B5CF6'
    },
    'KHOA HỌC CÔNG NGHỆ': {
        bg: 'bg-cyan-50',
        text: 'text-cyan-700',
        border: 'border-cyan-200',
        icon: 'fas fa-laptop-code',
        iconColor: 'text-cyan-500',
        progressColor: '#06B6D4'
    },
    'XE CỘ': {
        bg: 'bg-slate-50',
        text: 'text-slate-700',
        border: 'border-slate-200',
        icon: 'fas fa-car',
        iconColor: 'text-slate-500',
        progressColor: '#64748B'
    },
    'ĐỜI SỐNG': {
        bg: 'bg-pink-50',
        text: 'text-pink-700',
        border: 'border-pink-200',
        icon: 'fas fa-home',
        iconColor: 'text-pink-500',
        progressColor: '#EC4899'
    },
    'THẾ GIỚI': {
        bg: 'bg-teal-50',
        text: 'text-teal-700',
        border: 'border-teal-200',
        icon: 'fas fa-globe',
        iconColor: 'text-teal-500',
        progressColor: '#14B8A6'
    }
};

// DOM Elements
const newsInput = document.getElementById('newsInput');
const charCount = document.getElementById('charCount');
const clearBtn = document.getElementById('clearBtn');
const classifyBtn = document.getElementById('classifyBtn');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');

// Character counter
newsInput.addEventListener('input', () => {
    const length = newsInput.value.length;
    charCount.textContent = `${length} ký tự`;

    if (length > 5000) {
        charCount.classList.remove('bg-gray-100', 'text-gray-500');
        charCount.classList.add('bg-red-100', 'text-red-600');
    } else {
        charCount.classList.remove('bg-red-100', 'text-red-600');
        charCount.classList.add('bg-gray-100', 'text-gray-500');
    }
});

// Clear button
clearBtn.addEventListener('click', () => {
    newsInput.value = '';
    charCount.textContent = '0 ký tự';
    resultSection.classList.add('hidden');
    loadingSection.classList.add('hidden');
});

// Classify button
classifyBtn.addEventListener('click', async () => {
    const text = newsInput.value.trim();

    if (!text) {
        alert('Vui lòng nhập nội dung tin tức!');
        return;
    }

    // Show loading
    loadingSection.classList.remove('hidden');
    resultSection.classList.add('hidden');

    // Disable button
    classifyBtn.disabled = true;
    classifyBtn.classList.add('opacity-50', 'cursor-not-allowed');
    classifyBtn.innerHTML = `
        <div class="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
        <span>Đang xử lý...</span>
    `;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log("DEBUG: Received data from API", data);

        // Hide loading and show result
        setTimeout(() => {
            loadingSection.classList.add('hidden');
            displayResult(data);

            // Scroll to result
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 500);

    } catch (error) {
        console.error('Error:', error);
        loadingSection.classList.add('hidden');
        alert('Có lỗi xảy ra khi phân loại! Vui lòng thử lại.');
    } finally {
        // Re-enable button
        classifyBtn.disabled = false;
        classifyBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        classifyBtn.innerHTML = `
            <i class="fas fa-paper-plane"></i>
            <span>Phân loại</span>
        `;
    }
});

function displayResult(data) {
    const { label, confidence } = data;
    const config = CATEGORY_CONFIGS[label] || CATEGORY_CONFIGS['THỂ THAO'];

    // Update category card styling
    const categoryCard = document.getElementById('categoryCard');
    categoryCard.className = `rounded-2xl p-6 border flex flex-col items-center justify-center text-center h-full min-h-[250px] relative overflow-hidden transition-all hover:scale-[1.02] duration-300 ${config.bg} ${config.border} border-opacity-20`;

    // Update icon
    const categoryIcon = document.getElementById('categoryIcon');
    categoryIcon.className = `${config.icon} ${config.iconColor} text-5xl`;

    // Update label
    const categoryLabel = document.getElementById('categoryLabel');
    categoryLabel.className = `text-3xl font-black ${config.text}`;
    categoryLabel.textContent = label;

    // Update confidence with animation
    const confidenceValue = parseFloat(confidence);
    animateConfidence(confidenceValue, config.progressColor);

    // Show result section
    resultSection.classList.remove('hidden');

    // Update segmentation status
    const segmentationStatusText = document.getElementById('segmentationStatusText');
    if (segmentationStatusText) {
        segmentationStatusText.textContent = data.segmentation_status || "Không có thông tin";

        // Add color based on status
        if (data.segmentation_status && data.segmentation_status.includes("Đã tách từ")) {
            segmentationStatusText.className = "text-green-600 leading-relaxed text-lg font-medium";
        } else {
            segmentationStatusText.className = "text-red-600 leading-relaxed text-lg font-medium";
        }
    }

    // Render Explanation
    if (data.explanation) {
        console.log("DEBUG: Rendering explanation", data.explanation);
        renderExplanation(data.explanation);
    } else {
        console.warn("DEBUG: No explanation data found in response");
    }
}

function renderExplanation(wordsWithScores) {
    const container = document.getElementById('explanationContainer');
    container.innerHTML = '';

    // Find max score for normalization
    const maxScore = Math.max(...wordsWithScores.map(w => w.score));

    wordsWithScores.forEach(item => {
        const { word, score } = item;

        // Normalize score 0-1 relative to max score
        // Use a power function to make differences more visible
        const normalizedScore = Math.pow(score / maxScore, 0.7);

        const span = document.createElement('span');
        span.textContent = word + ' ';
        span.className = 'inline-block rounded px-1 mx-0.5 transition-all duration-300 hover:scale-110 cursor-default';

        // Calculate background color (Yellow/Orange)
        // We only highlight if score is significant enough
        if (normalizedScore > 0.1) {
            // R: 255, G: 215, B: 0 (Gold) -> R: 255, G: 165, B: 0 (Orange)
            // Alpha based on score
            const alpha = Math.min(normalizedScore * 0.8, 1.0); // Cap alpha
            span.style.backgroundColor = `rgba(255, 200, 0, ${alpha})`;

            // Tooltip for score
            span.title = `Độ quan trọng: ${(normalizedScore * 100).toFixed(1)}%`;
        }

        container.appendChild(span);
    });
}

function animateConfidence(targetValue, color) {
    const progressCircle = document.getElementById('progressCircle');
    const confidenceText = document.getElementById('confidenceText');

    // Set color
    progressCircle.setAttribute('stroke', color);

    // Calculate circumference
    const radius = 52;
    const circumference = 2 * Math.PI * radius;

    // Animate from 0 to target
    let current = 0;
    const increment = targetValue / 50; // 50 steps
    const interval = 20; // ms

    const timer = setInterval(() => {
        if (current >= targetValue) {
            current = targetValue;
            clearInterval(timer);
        }

        const offset = circumference - (current / 100) * circumference;
        progressCircle.style.strokeDashoffset = offset;
        confidenceText.textContent = `${Math.round(current)}%`;

        current += increment;
    }, interval);
}
