// Enhanced frontend JavaScript with improved functionality and optimization

// Global state management
const AppState = {
    config: {},
    videos: [],  // Initialize as empty array
    currentTaskId: null,
    progressInterval: null,
    isGenerating: false,
    currentPage: 1,
    videosPerPage: 12,
    searchQuery: '',
    sortOrder: 'newest',
    filterType: '',
    videoCache: new Map(),
    abortController: null
};

// DOM Elements cache
const DOM = {
    generateForm: null,
    progressSection: null,
    progressFill: null,
    progressStage: null,
    videoGallery: null,
    emptyGallery: null,
    filterPromptType: null,
    refreshGallery: null,
    videoModal: null,
    modalVideo: null,
    modalDetails: null,
    errorModal: null,
    errorMessage: null,
    searchInput: null,
    sortSelect: null,
    paginationContainer: null
};

// Initialize DOM elements
function initializeDOMElements() {
    DOM.generateForm = document.getElementById('generateForm');
    DOM.progressSection = document.getElementById('progressSection');
    DOM.progressFill = document.getElementById('progressFill');
    DOM.progressStage = document.getElementById('progressStage');
    DOM.videoGallery = document.getElementById('videoGallery');
    DOM.emptyGallery = document.getElementById('emptyGallery');
    DOM.filterPromptType = document.getElementById('filterPromptType');
    DOM.refreshGallery = document.getElementById('refreshGallery');
    DOM.videoModal = document.getElementById('videoModal');
    DOM.modalVideo = document.getElementById('modalVideo');
    DOM.modalDetails = document.getElementById('modalDetails');
    DOM.errorModal = document.getElementById('errorModal');
    DOM.errorMessage = document.getElementById('errorMessage');
    
    // Log missing elements for debugging
    Object.entries(DOM).forEach(([key, element]) => {
        if (!element && key !== 'searchInput' && key !== 'sortSelect' && key !== 'paginationContainer') {
            console.warn(`DOM element not found: ${key}`);
        }
    });
}

// Enhanced initialization
document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log('Initializing application...');
        
        // Initialize DOM elements first
        initializeDOMElements();
        
        // Set default empty arrays to prevent errors
        AppState.videos = [];
        
        // Load configuration
        await loadConfig();
        
        // Setup event listeners early
        setupEventListeners();
        setupKeyboardShortcuts();
        
        // Load videos
        await loadVideos();
        
        // Check server connection
        checkServerConnection();
        
        // Add enhanced controls after everything is loaded
        // Use setTimeout to ensure DOM is fully ready
        setTimeout(() => {
            try {
                addEnhancedControls();
            } catch (error) {
                console.error('Failed to add enhanced controls:', error);
            }
        }, 100);
        
        console.log('Application initialized successfully');
        
    } catch (error) {
        console.error('Initialization error:', error);
        showError('Failed to initialize application: ' + error.message);
    }
});

// Add enhanced controls to the UI
function addEnhancedControls() {
    try {
        const gallerySection = document.querySelector('.gallery-section');
        if (!gallerySection) {
            console.warn('Gallery section not found, cannot add enhanced controls.');
            return;
        }

        const galleryHeader = gallerySection.querySelector('.gallery-header');
        if (!galleryHeader) {
            console.warn('Gallery header not found, cannot add enhanced controls.');
            return;
        }

        let controls = galleryHeader.querySelector('.gallery-controls');
        if (!controls) {
            console.warn('Gallery controls not found, creating a new one.');
            controls = document.createElement('div');
            controls.className = 'gallery-controls';
            galleryHeader.appendChild(controls);
        }

        // Add search input
        if (!document.getElementById('searchVideos')) {
            const searchContainer = document.createElement('div');
            searchContainer.className = 'search-container';
            searchContainer.innerHTML = `<input type="text" id="searchVideos" placeholder="Search videos..." class="search-input">`;
            controls.prepend(searchContainer); // Prepend for better layout
            DOM.searchInput = document.getElementById('searchVideos');
            if (DOM.searchInput) {
                DOM.searchInput.addEventListener('input', debounce(handleSearch, 300));
            }
        }

        // Add sort select
        if (!document.getElementById('sortVideos')) {
            const sortContainer = document.createElement('div');
            sortContainer.innerHTML = `
                <select id="sortVideos" class="filter-select">
                    <option value="newest">Newest First</option>
                    <option value="oldest">Oldest First</option>
                    <option value="name">Name (A-Z)</option>
                    <option value="size">Size (Largest)</option>
                </select>
            `;
            controls.appendChild(sortContainer);
            DOM.sortSelect = document.getElementById('sortVideos');
            if (DOM.sortSelect) {
                DOM.sortSelect.addEventListener('change', handleSort);
            }
        }

        // Move existing filter and refresh button into the controls div for consistency
        if (DOM.filterPromptType && DOM.filterPromptType.parentElement !== controls) {
            controls.appendChild(DOM.filterPromptType);
        }
        if (DOM.refreshGallery && DOM.refreshGallery.parentElement !== controls) {
            controls.appendChild(DOM.refreshGallery);
        }

        // Add pagination container
        if (!document.getElementById('paginationContainer')) {
            const paginationDiv = document.createElement('div');
            paginationDiv.id = 'paginationContainer';
            paginationDiv.className = 'pagination-container';
            gallerySection.appendChild(paginationDiv);
            DOM.paginationContainer = paginationDiv;
        }
        
        console.log('Enhanced controls added successfully');
    } catch (error) {
        console.error('Error adding enhanced controls:', error);
    }
}

// Server connection check
async function checkServerConnection() {
    try {
        const response = await fetch('/api/config');
        if (!response.ok) {
            showNotification('Server connection issue detected', 'warning');
        }
    } catch (error) {
        showNotification('Cannot connect to server. Please check if the server is running.', 'error');
    }
}

// Enhanced configuration loading with error recovery
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        if (!response.ok) throw new Error('Failed to fetch configuration');
        
        AppState.config = await response.json();
        
        // Populate form dropdowns
        populateSelect('promptType', AppState.config.prompt_types);
        populateSelect('model', AppState.config.models);
        populateSelect('voice', AppState.config.voices);
        populateSelect('subtitleStyle', AppState.config.subtitle_styles);
        populateSelect('subtitleAnimation', AppState.config.subtitle_animations);
        populateSelect('subtitlePosition', AppState.config.subtitle_positions);
        populateSelect('transitionStyle', AppState.config.transition_styles);
        
        // Populate filter dropdown
        populateSelect('filterPromptType', AppState.config.prompt_types, true);
        
        // Load saved preferences
        loadUserPreferences();
        
    } catch (error) {
        showError('Failed to load configuration: ' + error.message);
        console.error('Config load error:', error);
        
        // Provide fallback configuration
        AppState.config = {
            prompt_types: ['did_you_know', 'conspiracy_theory', 'fake_history'],
            models: ['Flux', 'SD'],
            voices: ['raspy', 'upbeat', 'expressive'],
            subtitle_styles: ['modern', 'minimal', 'bold'],
            subtitle_animations: ['phrase', 'word', 'typewriter'],
            subtitle_positions: ['bottom', 'top', 'middle'],
            transition_styles: ['fade', 'slide', 'zoom']
        };
    }
}

// Load user preferences from localStorage
function loadUserPreferences() {
    try {
        const preferences = JSON.parse(localStorage.getItem('videoGenPreferences') || '{}');
        
        // Apply saved form values
        if (preferences.formData && DOM.generateForm) {
            Object.entries(preferences.formData).forEach(([key, value]) => {
                const element = document.querySelector(`[name="${key}"]`);
                if (element) {
                    if (element.type === 'checkbox') {
                        element.checked = value;
                    } else {
                        element.value = value;
                    }
                }
            });
        }
        
        // Apply saved view preferences
        if (preferences.videosPerPage) {
            AppState.videosPerPage = preferences.videosPerPage;
        }
    } catch (error) {
        console.error('Error loading preferences:', error);
    }
}

// Save user preferences
function saveUserPreferences() {
    try {
        if (!DOM.generateForm) return;
        
        const formData = new FormData(DOM.generateForm);
        const preferences = {
            formData: Object.fromEntries(formData),
            videosPerPage: AppState.videosPerPage
        };
        localStorage.setItem('videoGenPreferences', JSON.stringify(preferences));
    } catch (error) {
        console.error('Error saving preferences:', error);
    }
}

// Enhanced populate select with better formatting
function populateSelect(elementId, options, includeEmpty = false) {
    const select = document.getElementById(elementId);
    if (!select) return;
    
    select.innerHTML = '';
    
    if (includeEmpty) {
        const emptyOption = document.createElement('option');
        emptyOption.value = '';
        emptyOption.textContent = 'All Types';
        select.appendChild(emptyOption);
    }
    
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = formatOptionText(option);
        select.appendChild(optionElement);
    });
}

// Format option text for display
function formatOptionText(text) {
    if (!text) return 'Unknown';
    
    return String(text).split(/[_-]/).map(word => 
        word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    ).join(' ');
}

// Enhanced event listeners setup
function setupEventListeners() {
    // Form submission
    if (DOM.generateForm) {
        DOM.generateForm.addEventListener('submit', handleGenerate);
        // Form changes to save preferences
        DOM.generateForm.addEventListener('change', debounce(saveUserPreferences, 1000));
    }
    
    // Gallery controls
    if (DOM.refreshGallery) {
        DOM.refreshGallery.addEventListener('click', () => loadVideos(true));
    }
    
    if (DOM.filterPromptType) {
        DOM.filterPromptType.addEventListener('change', handleFilter);
    }
    
    // Modal controls
    const modalClose = document.querySelector('.modal-close');
    if (modalClose) {
        modalClose.addEventListener('click', closeVideoModal);
    }
    
    const errorClose = document.querySelector('.error-close');
    if (errorClose) {
        errorClose.addEventListener('click', closeErrorModal);
    }
    
    // Click outside modal to close
    if (DOM.videoModal) {
        DOM.videoModal.addEventListener('click', (e) => {
            if (e.target === DOM.videoModal) closeVideoModal();
        });
    }
    
    if (DOM.errorModal) {
        DOM.errorModal.addEventListener('click', (e) => {
            if (e.target === DOM.errorModal) closeErrorModal();
        });
    }
    
    // Advanced settings toggle animation
    const advancedSettings = document.querySelector('.advanced-settings');
    if (advancedSettings) {
        advancedSettings.addEventListener('toggle', (e) => {
            if (e.target.open) {
                e.target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    }
}

// Keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Escape to close modals
        if (e.key === 'Escape') {
            if (DOM.videoModal && DOM.videoModal.style.display !== 'none') closeVideoModal();
            if (DOM.errorModal && DOM.errorModal.style.display !== 'none') closeErrorModal();
        }
        
        // Ctrl/Cmd + Enter to generate
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !AppState.isGenerating) {
            DOM.generateForm.dispatchEvent(new Event('submit'));
        }
        
        // R to refresh gallery
        if (e.key === 'r' && !e.ctrlKey && !e.metaKey && 
            e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            loadVideos(true);
        }
    });
}

// Enhanced form submission with validation
async function handleGenerate(e) {
    e.preventDefault();
    
    if (AppState.isGenerating) {
        showNotification('Generation already in progress', 'warning');
        return;
    }
    
    // Get and validate form data
    const formData = new FormData(DOM.generateForm);
    const topic = formData.get('topic')?.trim();
    
    if (!topic) {
        showError('Please enter a topic');
        return;
    }
    
    if (topic.length < 3) {
        showError('Topic must be at least 3 characters long');
        return;
    }
    
    // Build request data
    const requestData = {
        topic: topic,
        prompt_type: formData.get('promptType'),
        model: formData.get('model'),
        voice: formData.get('voice'),
        fps: parseInt(formData.get('fps')),
        aspect_ratio: formData.get('aspectRatio').split(':').map(n => parseInt(n)),
        transition_duration: parseFloat(formData.get('transitionDuration')),
        pan_effect: formData.get('panEffect') === 'on',
        zoom_effect: formData.get('zoomEffect') === 'on',
        subtitles: formData.get('subtitles') === 'on',
        subtitle_style: formData.get('subtitleStyle'),
        subtitle_animation: formData.get('subtitleAnimation'),
        subtitle_position: formData.get('subtitlePosition'),
        highlight_keywords: formData.get('highlightKeywords') === 'on',
        transition_style: formData.get('transitionStyle')
    };
    
    try {
        // Set generating state
        AppState.isGenerating = true;
        setFormState(false);
        showGeneratingState(true);
        
        // Create abort controller for cancellation
        AppState.abortController = new AbortController();
        
        // Start generation
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData),
            signal: AppState.abortController.signal
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }
        
        const result = await response.json();
        AppState.currentTaskId = result.task_id;
        
        // Show progress section with animation
        DOM.progressSection.style.display = 'block';
        DOM.progressSection.classList.add('active');
        
        startProgressMonitoring();
        
    } catch (error) {
        if (error.name === 'AbortError') {
            showNotification('Generation cancelled', 'info');
        } else {
            showError('Failed to start generation: ' + error.message);
        }
        resetForm();
    }
}

// Cancel generation
function cancelGeneration() {
    if (AppState.abortController) {
        AppState.abortController.abort();
    }
    if (AppState.progressInterval) {
        clearInterval(AppState.progressInterval);
    }
    resetForm();
}

// Enhanced progress monitoring with better error handling
function startProgressMonitoring() {
    let failedAttempts = 0;
    const maxFailedAttempts = 3;
    
    AppState.progressInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${AppState.currentTaskId}`);
            
            if (!response.ok) {
                throw new Error('Failed to fetch status');
            }
            
            const status = await response.json();
            
            // Reset failed attempts on successful fetch
            failedAttempts = 0;
            
            // Update progress with smooth animation
            updateProgress(status.progress, status.stage);
            
            // Check completion status
            if (status.status === 'completed') {
                clearInterval(AppState.progressInterval);
                DOM.progressStage.textContent = '✓ Video generated successfully!';
                DOM.progressFill.style.background = 'var(--success)';
                
                showNotification('Video generated successfully!', 'success');
                
                setTimeout(() => {
                    resetForm();
                    loadVideos();
                }, 2000);
                
            } else if (status.status === 'failed') {
                clearInterval(AppState.progressInterval);
                showError('Generation failed: ' + (status.error || 'Unknown error'));
                resetForm();
            }
            
        } catch (error) {
            failedAttempts++;
            
            if (failedAttempts >= maxFailedAttempts) {
                clearInterval(AppState.progressInterval);
                showError('Lost connection to server. Please check if generation is still running.');
                resetForm();
            }
        }
    }, 2000);
}

// Update progress with smooth animation
function updateProgress(progress, stage) {
    // Smooth progress bar update
    DOM.progressFill.style.width = progress + '%';
    
    // Update stage with fade effect
    if (DOM.progressStage.textContent !== stage) {
        DOM.progressStage.style.opacity = '0';
        setTimeout(() => {
            DOM.progressStage.textContent = stage;
            DOM.progressStage.style.opacity = '1';
        }, 150);
    }
    
    // Add progress percentage to stage
    if (progress > 0 && progress < 100) {
        DOM.progressStage.textContent = `${stage} (${progress}%)`;
    }
}

// Set form state
function setFormState(enabled) {
    const elements = DOM.generateForm.querySelectorAll('input, select, button');
    elements.forEach(el => {
        el.disabled = !enabled;
        if (!enabled) {
            el.classList.add('disabled');
        } else {
            el.classList.remove('disabled');
        }
    });
}

// Show generating state
function showGeneratingState(show) {
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.querySelector('.btn-loader');
    
    if (show) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';
    } else {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Reset form after generation
function resetForm() {
    AppState.isGenerating = false;
    setFormState(true);
    showGeneratingState(false);
    
    DOM.progressSection.classList.remove('active');
    setTimeout(() => {
        DOM.progressSection.style.display = 'none';
    }, 300);
    
    DOM.progressFill.style.width = '0%';
    DOM.progressFill.style.background = '';
    DOM.progressStage.textContent = 'Initializing...';
    
    AppState.currentTaskId = null;
    AppState.abortController = null;
    
    if (AppState.progressInterval) {
        clearInterval(AppState.progressInterval);
        AppState.progressInterval = null;
    }
}

// Enhanced video loading with caching and error handling
async function loadVideos(forceRefresh = false) {
    try {
        // Show loading state
        if (DOM.videoGallery) {
            DOM.videoGallery.classList.add('loading');
        }
        
        const response = await fetch('/api/videos');
        if (!response.ok) throw new Error('Failed to load videos');
        
        const data = await response.json();
        console.log('Videos response:', data);

        // More robust data extraction logic
        let videosArray = [];
        if (data && typeof data === 'object' && Array.isArray(data.videos)) {
            videosArray = data.videos;
        } else if (Array.isArray(data)) {
            videosArray = data; // Fallback for APIs returning a raw array
        } else {
            console.warn('Unexpected video response format. Defaulting to empty array.', data);
        }
        AppState.videos = videosArray;
        
        console.log('AppState.videos is now:', AppState.videos);
        console.log('Is array?', Array.isArray(AppState.videos));
        
        // Sort videos by default
        sortVideos();
        
        // Update cache
        if (forceRefresh) {
            AppState.videoCache.clear();
        }
        
        renderGallery();
        
        if (forceRefresh) {
            showNotification('Gallery refreshed', 'success');
        }
        
    } catch (error) {
        console.error('Load videos error:', error);
        showError('Failed to load videos: ' + error.message);
        AppState.videos = [];
        renderGallery(); // Still render to show empty state
    } finally {
        if (DOM.videoGallery) {
            DOM.videoGallery.classList.remove('loading');
        }
    }
}

// Sort videos
function sortVideos() {
    if (!Array.isArray(AppState.videos)) {
        console.error('Cannot sort - AppState.videos is not an array:', AppState.videos);
        AppState.videos = [];
        return;
    }
    
    if (AppState.videos.length === 0) {
        return;
    }
    
    const sortOrder = DOM.sortSelect?.value || AppState.sortOrder;
    
    try {
        AppState.videos.sort((a, b) => {
            if (!a || !b) return 0;
            
            switch (sortOrder) {
                case 'newest':
                    return new Date(b.created_at || 0) - new Date(a.created_at || 0);
                case 'oldest':
                    return new Date(a.created_at || 0) - new Date(b.created_at || 0);
                case 'name':
                    return (a.topic || '').localeCompare(b.topic || '');
                case 'size':
                    return (b.size || 0) - (a.size || 0);
                default:
                    return 0;
            }
        });
    } catch (error) {
        console.error('Error sorting videos:', error);
    }
}

// Search handler
function handleSearch(e) {
    AppState.searchQuery = e.target.value.toLowerCase();
    AppState.currentPage = 1;
    renderGallery();
}

// Sort handler
function handleSort(e) {
    AppState.sortOrder = e.target.value;
    sortVideos();
    renderGallery();
}

// Filter handler
function handleFilter(e) {
    AppState.filterType = e.target.value;
    AppState.currentPage = 1;
    renderGallery();
}

// Get filtered and searched videos
function getFilteredVideos() {
    if (!Array.isArray(AppState.videos)) {
        console.error('AppState.videos is not an array, resetting to empty array');
        AppState.videos = [];
        return [];
    }
    
    let filtered = AppState.videos;
    
    // Apply type filter
    if (AppState.filterType) {
        filtered = filtered.filter(video => video.prompt_type === AppState.filterType);
    }
    
    // Apply search filter
    if (AppState.searchQuery) {
        filtered = filtered.filter(video => 
            (video.topic || '').toLowerCase().includes(AppState.searchQuery) ||
            (video.model || '').toLowerCase().includes(AppState.searchQuery) ||
            (video.voice || '').toLowerCase().includes(AppState.searchQuery)
        );
    }
    
    return filtered;
}

// Enhanced gallery rendering with pagination
function renderGallery() {
    if (!DOM.videoGallery || !DOM.emptyGallery) {
        console.error('Gallery DOM elements not found');
        return;
    }
    
    const filteredVideos = getFilteredVideos();
    const totalPages = Math.ceil(filteredVideos.length / AppState.videosPerPage);
    
    if (AppState.currentPage > totalPages) {
        AppState.currentPage = totalPages || 1;
    }
    
    const startIndex = (AppState.currentPage - 1) * AppState.videosPerPage;
    const endIndex = startIndex + AppState.videosPerPage;
    const pageVideos = filteredVideos.slice(startIndex, endIndex);
    
    if (pageVideos.length === 0) {
        DOM.videoGallery.style.display = 'none';
        DOM.emptyGallery.style.display = 'block';
        if (DOM.paginationContainer) {
            DOM.paginationContainer.innerHTML = '';
        }
        return;
    }
    
    DOM.videoGallery.style.display = 'grid';
    DOM.emptyGallery.style.display = 'none';
    
    const videoCards = pageVideos.map(video => createVideoCard(video)).join('');
    DOM.videoGallery.innerHTML = videoCards;
    
    const cards = DOM.videoGallery.querySelectorAll('.video-card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 50}ms`;
        card.classList.add('fade-in');
    });
    
    if (DOM.paginationContainer) {
        renderPagination(totalPages);
    }
    
    updateResultsInfo(filteredVideos.length);
}

// Create video card HTML
function createVideoCard(video) {
    const videoId = video.id || 'unknown';
    const thumbnail = AppState.videoCache.get(videoId) || '/api/video/' + videoId + '/thumbnail';
    const duration = video.duration ? formatDuration(video.duration) : '';
    const topic = video.topic || 'Untitled Video';
    const size = video.size || 0;
    const created = video.created_at || '';
    const status = video.status || 'completed';
    
    return `
        <div class="video-card" data-id="${videoId}">
            <div class="video-thumbnail">
                <img src="${thumbnail}" 
                     alt="${escapeHtml(topic)}"
                     loading="lazy"
                     onerror="this.style.display='none'; this.parentElement.innerHTML='🎬'">
                ${duration ? `<span class="video-duration">${duration}</span>` : ''}
            </div>
            <div class="video-info">
                <div class="video-title" title="${escapeHtml(topic)}">${escapeHtml(topic)}</div>
                <div class="video-meta">
                    <span><span class="status-indicator ${status}"></span>${formatOptionText(video.prompt_type)}</span>
                    <span>📊 ${video.model || 'Unknown'} | 🎤 ${formatOptionText(video.voice)}</span>
                    <span>📅 ${formatDate(created)}</span>
                    <span>💾 ${formatFileSize(size)}</span>
                </div>
                <div class="video-actions">
                    <button class="play-btn" onclick="playVideo('${videoId}')" title="Play video">
                        <span>▶ Play</span>
                    </button>
                    <button class="download-btn" onclick="downloadVideo('${videoId}')" title="Download video">
                        <span>⬇ Download</span>
                    </button>
                    <button class="delete-btn" onclick="deleteVideo('${videoId}')" title="Delete video">
                        <span>🗑 Delete</span>
                    </button>
                </div>
            </div>
        </div>
    `;
}

// Render pagination controls
function renderPagination(totalPages) {
    if (!DOM.paginationContainer) {
        console.warn('Pagination container not found');
        return;
    }
    
    if (totalPages <= 1) {
        DOM.paginationContainer.innerHTML = '';
        return;
    }
    
    let paginationHTML = '<div class="pagination">';
    
    paginationHTML += `
        <button class="pagination-btn" onclick="changePage(${AppState.currentPage - 1})" 
                ${AppState.currentPage === 1 ? 'disabled' : ''}>
            Previous
        </button>
    `;
    
    const maxVisiblePages = 5;
    let startPage = Math.max(1, AppState.currentPage - Math.floor(maxVisiblePages / 2));
    let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
    
    if (endPage - startPage < maxVisiblePages - 1) {
        startPage = Math.max(1, endPage - maxVisiblePages + 1);
    }
    
    if (startPage > 1) {
        paginationHTML += `<button class="pagination-btn" onclick="changePage(1)">1</button>`;
        if (startPage > 2) {
            paginationHTML += `<span class="pagination-ellipsis">...</span>`;
        }
    }
    
    for (let i = startPage; i <= endPage; i++) {
        paginationHTML += `
            <button class="pagination-btn ${i === AppState.currentPage ? 'active' : ''}" 
                    onclick="changePage(${i})">${i}</button>
        `;
    }
    
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            paginationHTML += `<span class="pagination-ellipsis">...</span>`;
        }
        paginationHTML += `<button class="pagination-btn" onclick="changePage(${totalPages})">${totalPages}</button>`;
    }
    
    paginationHTML += `
        <button class="pagination-btn" onclick="changePage(${AppState.currentPage + 1})" 
                ${AppState.currentPage === totalPages ? 'disabled' : ''}>
            Next
        </button>
    `;
    
    paginationHTML += '</div>';
    
    DOM.paginationContainer.innerHTML = paginationHTML;
}

// Change page
window.changePage = function(page) {
    AppState.currentPage = page;
    renderGallery();
    
    if (DOM.videoGallery) {
        DOM.videoGallery.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
};

// Update results info
function updateResultsInfo(totalResults) {
    const galleryHeader = document.querySelector('.gallery-header');
    if (!galleryHeader) return;
    
    let info = galleryHeader.querySelector('.results-info');
    if (!info) {
        info = document.createElement('div');
        info.className = 'results-info';
        galleryHeader.appendChild(info);
    }
    
    const start = (AppState.currentPage - 1) * AppState.videosPerPage + 1;
    const end = Math.min(start + AppState.videosPerPage - 1, totalResults);
    
    info.textContent = `Showing ${start}-${end} of ${totalResults} videos`;
}


// Play video in modal with enhanced features
window.playVideo = async function(videoId) {
    try {
        const video = AppState.videos.find(v => v.id === videoId);
        if (!video) {
            showError('Video not found');
            return;
        }
        
        DOM.videoModal.style.display = 'flex';
        DOM.modalVideo.style.display = 'none';
        DOM.modalDetails.innerHTML = '<div class="loading-spinner">Loading video...</div>';
        
        DOM.modalVideo.src = `/api/video/${videoId}`;
        
        await new Promise((resolve, reject) => {
            DOM.modalVideo.onloadedmetadata = resolve;
            DOM.modalVideo.onerror = reject;
            setTimeout(() => reject(new Error('Video load timeout')), 10000);
        });
        
        DOM.modalVideo.style.display = 'block';
        
        const detailsHTML = `
            <h3>${escapeHtml(video.topic)}</h3>
            <div class="video-details-grid">
                <div class="detail-item">
                    <strong>Type:</strong> ${formatOptionText(video.prompt_type || 'Unknown')}
                </div>
                <div class="detail-item">
                    <strong>Model:</strong> ${video.model || 'Unknown'}
                </div>
                <div class="detail-item">
                    <strong>Voice:</strong> ${formatOptionText(video.voice || 'Unknown')}
                </div>
                <div class="detail-item">
                    <strong>Created:</strong> ${formatDate(video.created_at, true)}
                </div>
                <div class="detail-item">
                    <strong>Size:</strong> ${formatFileSize(video.size || 0)}
                </div>
                <div class="detail-item">
                    <strong>Duration:</strong> ${formatDuration(DOM.modalVideo.duration)}
                </div>
            </div>
            
            ${video.video_params ? `
                <h4>Video Configuration</h4>
                <ul class="config-list">
                    <li>FPS: ${video.video_params.fps || 'N/A'}</li>
                    <li>Aspect Ratio: ${video.video_params.aspect_ratio?.join(':') || 'N/A'}</li>
                    <li>Subtitles: ${video.video_params.subtitles ? 'Yes' : 'No'}</li>
                    ${video.video_params.subtitles ? `
                        <li>Subtitle Style: ${formatOptionText(video.video_params.subtitle_style || 'N/A')}</li>
                        <li>Subtitle Animation: ${formatOptionText(video.video_params.subtitle_animation || 'N/A')}</li>
                    ` : ''}
                    <li>Effects: ${getEffectsList(video.video_params)}</li>
                    <li>Transition: ${formatOptionText(video.video_params.transition_style || 'fade')}</li>
                </ul>
            ` : ''}
            
            <div class="video-actions-modal">
                <button class="download-btn" onclick="downloadVideo('${videoId}')">
                    ⬇ Download Video
                </button>
                <button class="copy-btn" onclick="copyVideoLink('${videoId}')">
                    🔗 Copy Link
                </button>
            </div>
        `;
        
        DOM.modalDetails.innerHTML = detailsHTML;
        
        DOM.modalVideo.play().catch(e => {
            console.warn('Auto-play failed:', e);
        });
        
    } catch (error) {
        DOM.modalDetails.innerHTML = `
            <div class="error-message">
                <h3>Error Loading Video</h3>
                <p>Failed to load video. Please try again.</p>
            </div>
        `;
        console.error('Error loading video:', error);
    }
};

// Get effects list
function getEffectsList(params) {
    if (!params) return 'None';
    
    const effects = [];
    if (params.pan_effect) effects.push('Pan');
    if (params.zoom_effect) effects.push('Zoom');
    if (params.highlight_keywords) effects.push('Keyword Highlighting');
    return effects.length > 0 ? effects.join(', ') : 'None';
}

// Copy video link
window.copyVideoLink = async function(videoId) {
    const url = `${window.location.origin}/api/video/${videoId}`;
    
    try {
        await navigator.clipboard.writeText(url);
        showNotification('Video link copied to clipboard!', 'success');
    } catch (error) {
        showNotification('Failed to copy link', 'error');
    }
};

// Enhanced download with progress tracking
window.downloadVideo = async function(videoId) {
    try {
        const video = AppState.videos.find(v => v.id === videoId);
        if (!video) {
            showError('Video not found');
            return;
        }
        
        const a = document.createElement('a');
        a.href = `/api/video/${videoId}`;
        a.download = `${(video.topic || 'video').replace(/[^a-z0-9]/gi, '_').toLowerCase()}.mp4`;
        
        showNotification('Download started...', 'info');
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
    } catch (error) {
        showError('Failed to download video: ' + error.message);
    }
};

// Enhanced delete with confirmation
window.deleteVideo = async function(videoId) {
    try {
        const video = AppState.videos.find(v => v.id === videoId);
        if (!video) {
            showError('Video not found');
            return;
        }
        
        const confirmed = await showConfirmDialog(
            'Delete Video',
            `Are you sure you want to delete "${video.topic || 'this video'}"? This action cannot be undone.`
        );
        
        if (!confirmed) return;
        
        const response = await fetch(`/api/video/${videoId}`, { method: 'DELETE' });
        
        if (!response.ok) {
            throw new Error('Failed to delete video');
        }
        
        AppState.videos = AppState.videos.filter(v => v.id !== videoId);
        AppState.videoCache.delete(videoId);
        
        renderGallery();
        
        showNotification('Video deleted successfully', 'success');
        
        if (DOM.modalVideo && DOM.modalVideo.src.includes(videoId)) {
            closeVideoModal();
        }
        
    } catch (error) {
        showError('Failed to delete video: ' + error.message);
    }
};

// Close video modal
function closeVideoModal() {
    DOM.videoModal.style.display = 'none';
    DOM.modalVideo.pause();
    DOM.modalVideo.src = '';
    DOM.modalDetails.innerHTML = '';
}

// Show error modal
function showError(message) {
    console.error('Error:', message);
    
    if (!DOM.errorMessage || !DOM.errorModal) {
        console.error('Error modal elements not found, showing alert instead');
        alert('Error: ' + message);
        return;
    }
    
    DOM.errorMessage.textContent = message;
    DOM.errorModal.style.display = 'flex';
}

// Close error modal
function closeErrorModal() {
    DOM.errorModal.style.display = 'none';
    DOM.errorMessage.textContent = '';
}

// Show notification toast
function showNotification(message, type = 'info') {
    if (!document.body) {
        console.log(`[${type.toUpperCase()}] ${message}`);
        return;
    }
    
    let container = document.querySelector('.notification-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'notification-container';
        document.body.appendChild(container);
    }
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span class="notification-icon">${getNotificationIcon(type)}</span>
        <span class="notification-message">${escapeHtml(message)}</span>
    `;
    
    container.appendChild(notification);
    
    setTimeout(() => notification.classList.add('show'), 10);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, 4000);
}

// Get notification icon
function getNotificationIcon(type) {
    const icons = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ'
    };
    return icons[type] || icons.info;
}

// Show confirm dialog
async function showConfirmDialog(title, message) {
    return new Promise((resolve) => {
        const dialog = document.createElement('div');
        dialog.className = 'confirm-dialog-overlay';
        dialog.innerHTML = `
            <div class="confirm-dialog">
                <h3>${escapeHtml(title)}</h3>
                <p>${escapeHtml(message)}</p>
                <div class="confirm-actions">
                    <button class="confirm-btn cancel">Cancel</button>
                    <button class="confirm-btn delete">Delete</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(dialog);
        
        setTimeout(() => dialog.classList.add('show'), 10);
        
        const handleAction = (confirmed) => {
            dialog.classList.remove('show');
            setTimeout(() => {
                dialog.remove();
                resolve(confirmed);
            }, 300);
        };
        
        dialog.querySelector('.cancel').onclick = () => handleAction(false);
        dialog.querySelector('.delete').onclick = () => handleAction(true);
        dialog.onclick = (e) => {
            if (e.target === dialog) handleAction(false);
        };
    });
}

// Utility functions
function escapeHtml(text) {
    if (!text) return '';
    
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function formatDate(dateString, detailed = false) {
    if (!dateString || dateString === 'Unknown') return 'Unknown';
    
    try {
        const date = new Date(dateString);
        
        if (isNaN(date.getTime())) {
            return 'Unknown';
        }
        
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);
        
        if (!detailed && diffMins < 1) return 'Just now';
        if (!detailed && diffMins < 60) return `${diffMins}m ago`;
        if (!detailed && diffHours < 24) return `${diffHours}h ago`;
        if (!detailed && diffDays < 7) return `${diffDays}d ago`;
        
        if (detailed) {
            return date.toLocaleString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        }
        
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    } catch (error) {
        return 'Unknown';
    }
}

function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '0 Bytes';
    
    try {
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    } catch (error) {
        return '0 Bytes';
    }
}

function formatDuration(seconds) {
    if (!seconds || seconds === 0 || isNaN(seconds)) return '0:00';
    
    try {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    } catch (error) {
        return '0:00';
    }
}

// Debounce function for performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add CSS for new elements
const style = document.createElement('style');
style.textContent = `
    .notification-container {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 3000;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    .notification {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 12px 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: var(--shadow-lg);
        transform: translateX(400px);
        transition: transform 0.3s ease;
        max-width: 400px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification.success {
        border-color: var(--success);
        background: rgba(16, 185, 129, 0.1);
    }
    
    .notification.error {
        border-color: var(--error);
        background: rgba(239, 68, 68, 0.1);
    }
    
    .notification.warning {
        border-color: var(--warning);
        background: rgba(245, 158, 11, 0.1);
    }
    
    .notification.info {
        border-color: var(--info);
        background: rgba(59, 130, 246, 0.1);
    }
    
    .notification-icon {
        font-size: 1.2em;
    }
    
    .confirm-dialog-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(5px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 4000;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .confirm-dialog-overlay.show {
        opacity: 1;
    }
    
    .confirm-dialog {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 24px;
        max-width: 400px;
        box-shadow: var(--shadow-lg);
        transform: scale(0.9);
        transition: transform 0.3s ease;
    }
    
    .confirm-dialog-overlay.show .confirm-dialog {
        transform: scale(1);
    }
    
    .confirm-dialog h3 {
        margin-bottom: 12px;
        color: var(--text-primary);
    }
    
    .confirm-dialog p {
        margin-bottom: 20px;
        color: var(--text-secondary);
    }
    
    .confirm-actions {
        display: flex;
        gap: 12px;
        justify-content: flex-end;
    }
    
    .confirm-btn {
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all var(--transition-fast);
    }
    
    .confirm-btn.cancel {
        background: var(--bg-tertiary);
        color: var(--text-primary);
    }
    
    .confirm-btn.delete {
        background: var(--error);
        color: white;
    }
    
    .confirm-btn:hover {
        transform: scale(1.05);
    }
    
    .search-container {
        flex: 1;
        max-width: 300px;
    }
    
    .search-input {
        width: 100%;
        padding: 10px 16px;
        background: var(--bg-tertiary);
        border: 2px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-primary);
        font-size: 0.95em;
        transition: all var(--transition-base);
    }
    
    .search-input:focus {
        outline: none;
        border-color: var(--accent-primary);
        box-shadow: 0 0 0 3px var(--accent-glow);
    }
    
    .pagination-container {
        margin-top: 30px;
        display: flex;
        justify-content: center;
    }
    
    .pagination {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    
    .pagination-btn {
        padding: 8px 16px;
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        color: var(--text-primary);
        cursor: pointer;
        transition: all var(--transition-fast);
        font-weight: 500;
    }
    
    .pagination-btn:hover:not(:disabled) {
        background: var(--bg-hover);
        border-color: var(--accent-primary);
    }
    
    .pagination-btn.active {
        background: var(--accent-primary);
        color: white;
        border-color: var(--accent-primary);
    }
    
    .pagination-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .pagination-ellipsis {
        color: var(--text-tertiary);
        padding: 0 8px;
    }
    
    .results-info {
        font-size: 0.9em;
        color: var(--text-secondary);
        margin-left: auto;
    }
    
    .loading-spinner {
        text-align: center;
        padding: 40px;
        color: var(--text-secondary);
    }
    
    .video-details-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin: 20px 0;
        padding: 16px;
        background: var(--bg-tertiary);
        border-radius: 8px;
    }
    
    .detail-item {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    
    .detail-item strong {
        color: var(--text-secondary);
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .config-list {
        list-style: none;
        padding: 16px;
        background: var(--bg-tertiary);
        border-radius: 8px;
        margin: 12px 0;
    }
    
    .config-list li {
        padding: 4px 0;
        color: var(--text-secondary);
    }
    
    .video-actions-modal {
        display: flex;
        gap: 12px;
        margin-top: 20px;
    }
    
    .video-actions-modal button {
        flex: 1;
        padding: 12px 20px;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all var(--transition-fast);
    }
    
    .copy-btn {
        background: var(--bg-tertiary);
        color: var(--accent-primary);
        border: 2px solid var(--accent-primary);
    }
    
    .copy-btn:hover {
        background: var(--accent-primary);
        color: white;
    }
    
    .video-duration {
        position: absolute;
        bottom: 8px;
        right: 8px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    @media (max-width: 768px) {
        .notification-container {
            left: 20px;
            right: 20px;
        }
        
        .notification {
            max-width: none;
        }
        
        .search-container {
            max-width: none;
        }
        
        .pagination-btn {
            padding: 6px 12px;
            font-size: 0.9em;
        }
    }
`;
document.head.appendChild(style);