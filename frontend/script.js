// Optimized frontend JavaScript with Queue Support
const state = {
    config: {},
    videos: [],
    currentTaskId: null,
    progressInterval: null,
    isGenerating: false,
    currentPage: 1,
    videosPerPage: 12,
    searchQuery: '',
    filterType: '',
    // Queue state
    queue: [],
    queueProcessing: false,
    currentQueueIndex: -1,
    queueStats: {
        total: 0,
        completed: 0,
        failed: 0
    }
};

const $ = id => document.getElementById(id);
const $$ = (sel, ctx = document) => ctx.querySelector(sel);
const $$$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

const api = {
    async get(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${res.status}: ${res.statusText}`);
        return res.json();
    },
    async post(url, data) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!res.ok) throw new Error(`${res.status}: ${res.statusText}`);
        return res.json();
    },
    async delete(url) {
        const res = await fetch(url, { method: 'DELETE' });
        if (!res.ok) throw new Error(`${res.status}: ${res.statusText}`);
        return res.json();
    }
};

const utils = {
    formatText: text => {
        if (!text) return 'Unknown';
        const words = String(text).split(/[_-]/);
        let result = '';
        for (let i = 0; i < words.length; i++) {
            if (i > 0) result += ' ';
            result += words[i].charAt(0).toUpperCase() + words[i].slice(1);
        }
        return result;
    },
    formatDate: (date, detailed = false) => {
        if (!date) return 'Unknown';
        const d = new Date(date);
        if (isNaN(d)) return 'Unknown';
        
        const now = new Date();
        const diff = now - d;
        const mins = Math.floor(diff / 60000);
        const hours = Math.floor(mins / 60);
        const days = Math.floor(hours / 24);
        
        if (!detailed) {
            if (mins < 1) return 'Just now';
            if (mins < 60) return `${mins}m ago`;
            if (hours < 24) return `${hours}h ago`;
            if (days < 7) return `${days}d ago`;
        }
        
        return d.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            ...(detailed && { hour: '2-digit', minute: '2-digit' })
        });
    },
    formatSize: bytes => {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
    },
    formatDuration: sec => {
        if (!sec) return '0:00';
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    },
    formatTime: seconds => {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        return `${m}:${s.toString().padStart(2, '0')}`;
    },
    escapeHtml: text => {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    },
    debounce: (fn, wait) => {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn(...args), wait);
        };
    },
    generateId: () => `queue-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
};

const ui = {
    populateSelect(id, options, includeEmpty = false) {
        const sel = $(id);
        if (!sel) return;
        
        sel.innerHTML = includeEmpty ? '<option value="">All Types</option>' : '';
        for (let i = 0; i < options.length; i++) {
            const opt = options[i];
            const el = document.createElement('option');
            el.value = opt;
            el.textContent = utils.formatText(opt);
            sel.appendChild(el);
        }
    },
    
    showNotification(msg, type = 'info') {
        const container = $('notificationContainer');
        const notif = document.createElement('div');
        notif.className = `notification ${type}`;
        notif.innerHTML = `
            <span>${{success:'✓',error:'✗',warning:'⚠',info:'ℹ'}[type]||'ℹ'}</span>
            <span>${utils.escapeHtml(msg)}</span>
        `;
        
        container.appendChild(notif);
        setTimeout(() => notif.classList.add('show'), 10);
        setTimeout(() => {
            notif.classList.remove('show');
            setTimeout(() => notif.remove(), 300);
        }, 4000);
    },
    
    async confirm(title, msg) {
        return new Promise(resolve => {
            const overlay = document.createElement('div');
            overlay.className = 'modal';
            overlay.innerHTML = `
                <div class="modal-content" style="max-width: 400px;">
                    <h3>${utils.escapeHtml(title)}</h3>
                    <p style="margin: 16px 0;">${utils.escapeHtml(msg)}</p>
                    <div style="display: flex; gap: 12px; justify-content: flex-end;">
                        <button class="btn-cancel" style="padding: 10px 20px; background: var(--bg-tertiary); border: none; border-radius: 8px; cursor: pointer;">Cancel</button>
                        <button class="btn-confirm" style="padding: 10px 20px; background: var(--error); color: white; border: none; border-radius: 8px; cursor: pointer;">Confirm</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(overlay);
            overlay.style.display = 'flex';
            
            const cleanup = (result) => {
                overlay.remove();
                resolve(result);
            };
            
            $$('.btn-cancel', overlay).onclick = () => cleanup(false);
            $$('.btn-confirm', overlay).onclick = () => cleanup(true);
            overlay.onclick = e => e.target === overlay && cleanup(false);
        });
    },
    
    setFormState(enabled) {
        const form = $('generateForm');
        if (!form) return;
        
        const elements = $$$('input, select, button', form);
        elements.forEach(el => {
            el.disabled = !enabled;
            el.classList.toggle('disabled', !enabled);
        });
        
        const btnText = $$('.btn-text', form);
        const btnLoader = $$('.btn-loader', form);
        if (btnText) btnText.style.display = enabled ? 'inline' : 'none';
        if (btnLoader) btnLoader.style.display = enabled ? 'none' : 'inline-flex';
    },
    
    updateProgress(progress, stage) {
        const progressFill = $('progressFill');
        const progressStage = $('progressStage');
        if (progressFill) progressFill.style.width = progress + '%';
        if (progressStage) progressStage.textContent = stage + (progress > 0 && progress < 100 ? ` (${progress}%)` : '');
    }
};

// Queue Management Functions
const queue = {
    add(data) {
        const queueItem = {
            id: utils.generateId(),
            ...data,
            status: 'pending',
            addedAt: new Date().toISOString()
        };
        
        state.queue.push(queueItem);
        this.save();
        this.render();
        this.updateStats();
        
        ui.showNotification(`Added "${data.topic}" to queue`, 'success');
        return queueItem;
    },
    
    remove(id) {
        const index = state.queue.findIndex(item => item.id === id);
        if (index !== -1) {
            const item = state.queue[index];
            if (item.status === 'processing') {
                ui.showNotification('Cannot remove item currently being processed', 'warning');
                return;
            }
            state.queue.splice(index, 1);
            this.save();
            this.render();
            this.updateStats();
        }
    },
    
    clear() {
        if (state.queueProcessing) {
            ui.showNotification('Cannot clear queue while processing', 'warning');
            return;
        }
        state.queue = [];
        state.queueStats = { total: 0, completed: 0, failed: 0 };
        this.save();
        this.render();
        this.updateStats();
    },
    
    save() {
        localStorage.setItem('videoGenQueue', JSON.stringify(state.queue));
    },
    
    load() {
        const saved = localStorage.getItem('videoGenQueue');
        if (saved) {
            try {
                state.queue = JSON.parse(saved);
                // Reset any 'processing' status to 'pending'
                state.queue.forEach(item => {
                    if (item.status === 'processing') item.status = 'pending';
                });
            } catch (e) {
                console.error('Failed to load queue:', e);
                state.queue = [];
            }
        }
    },
    
    render() {
        const queueList = $('queueList');
        const emptyQueue = $('emptyQueue');
        const queueSection = $('queueSection');
        
        if (!queueList || !emptyQueue) return;
        
        if (state.queue.length === 0) {
            queueList.style.display = 'none';
            emptyQueue.style.display = 'block';
            $('startQueueBtn').disabled = true;
            $('clearQueueBtn').disabled = true;
        } else {
            queueList.style.display = 'block';
            emptyQueue.style.display = 'none';
            $('startQueueBtn').disabled = state.queueProcessing;
            $('clearQueueBtn').disabled = state.queueProcessing;
            
            queueList.innerHTML = state.queue.map((item, index) => `
                <div class="queue-item ${item.status}" data-id="${item.id}">
                    <div class="queue-item-number">${index + 1}</div>
                    <div class="queue-item-info">
                        <div class="queue-item-title">${utils.escapeHtml(item.topic)}</div>
                        <div class="queue-item-meta">
                            ${utils.formatText(item.prompt_type)} | ${item.model} | ${utils.formatText(item.voice)}
                        </div>
                    </div>
                    <div class="queue-item-status">
                        ${this.getStatusIcon(item.status)}
                        <span>${this.getStatusText(item.status)}</span>
                    </div>
                    ${item.status === 'pending' && !state.queueProcessing ? `
                        <button class="queue-item-remove" onclick="removeFromQueue('${item.id}')">×</button>
                    ` : ''}
                </div>
            `).join('');
        }
    },
    
    getStatusIcon(status) {
        const icons = {
            pending: '⏳',
            processing: '🔄',
            completed: '✅',
            failed: '❌'
        };
        return icons[status] || '❓';
    },
    
    getStatusText(status) {
        const texts = {
            pending: 'Waiting',
            processing: 'Processing',
            completed: 'Completed',
            failed: 'Failed'
        };
        return texts[status] || 'Unknown';
    },
    
    updateStats() {
        const stats = {
            total: state.queue.length,
            completed: state.queue.filter(i => i.status === 'completed').length,
            failed: state.queue.filter(i => i.status === 'failed').length
        };
        
        state.queueStats = stats;
        
        $('queueTotal').textContent = stats.total;
        $('queueCompleted').textContent = stats.completed;
        $('queueFailed').textContent = stats.failed;
        
        // Estimate time (90 seconds per video average)
        const remaining = stats.total - stats.completed - stats.failed;
        const estimatedSeconds = remaining * 90;
        $('queueTime').textContent = remaining > 0 ? utils.formatTime(estimatedSeconds) : '--:--';
        
        $('queueStats').style.display = stats.total > 0 ? 'flex' : 'none';
    },
    
    async process() {
        if (state.queueProcessing) return;
        
        const pendingItems = state.queue.filter(i => i.status === 'pending');
        if (pendingItems.length === 0) {
            ui.showNotification('No pending items in queue', 'info');
            return;
        }
        
        state.queueProcessing = true;
        ui.showNotification(`Starting queue processing (${pendingItems.length} videos)`, 'info');
        
        // Update UI
        const startBtn = $('startQueueBtn');
        const btnText = $$('.btn-text', startBtn);
        const btnLoader = $$('.btn-loader', startBtn);
        if (btnText) btnText.style.display = 'none';
        if (btnLoader) btnLoader.style.display = 'inline-flex';
        startBtn.disabled = true;
        $('clearQueueBtn').disabled = true;
        
        // Process each item
        for (let i = 0; i < state.queue.length; i++) {
            if (state.queue[i].status !== 'pending') continue;
            
            state.currentQueueIndex = i;
            const item = state.queue[i];
            
            // Update status
            item.status = 'processing';
            this.save();
            this.render();
            
            try {
                // Generate video
                await this.processItem(item);
                
                // Mark as completed
                item.status = 'completed';
                item.completedAt = new Date().toISOString();
                ui.showNotification(`✓ Completed: ${item.topic}`, 'success');
                
            } catch (error) {
                // Mark as failed
                item.status = 'failed';
                item.error = error.message;
                item.failedAt = new Date().toISOString();
                ui.showNotification(`✗ Failed: ${item.topic} - ${error.message}`, 'error');
            }
            
            this.save();
            this.render();
            this.updateStats();
            
            // Small delay between videos
            if (i < state.queue.length - 1) {
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
        
        // Complete
        state.queueProcessing = false;
        state.currentQueueIndex = -1;
        
        if (btnText) btnText.style.display = 'inline';
        if (btnLoader) btnLoader.style.display = 'none';
        startBtn.disabled = false;
        $('clearQueueBtn').disabled = false;
        
        const completed = state.queue.filter(i => i.status === 'completed').length;
        const failed = state.queue.filter(i => i.status === 'failed').length;
        
        ui.showNotification(
            `Queue processing complete! ✓ ${completed} successful, ✗ ${failed} failed`,
            failed > 0 ? 'warning' : 'success'
        );
        
        // Refresh gallery
        await loadVideos();
    },
    
    async processItem(item) {
        return new Promise(async (resolve, reject) => {
            try {
                const result = await api.post('/api/generate', {
                    topic: item.topic,
                    prompt_type: item.prompt_type,
                    model: item.model,
                    voice: item.voice,
                    fps: item.fps,
                    aspect_ratio: item.aspect_ratio,
                    transition_duration: item.transition_duration,
                    pan_effect: item.pan_effect,
                    zoom_effect: item.zoom_effect,
                    subtitles: item.subtitles,
                    subtitle_style: item.subtitle_style,
                    subtitle_animation: item.subtitle_animation,
                    highlight_keywords: item.highlight_keywords
                });
                
                const taskId = result.task_id;
                
                // Poll for completion
                const checkInterval = setInterval(async () => {
                    try {
                        const status = await api.get(`/api/status/${taskId}`);
                        
                        if (status.status === 'completed') {
                            clearInterval(checkInterval);
                            resolve();
                        } else if (status.status === 'failed') {
                            clearInterval(checkInterval);
                            reject(new Error(status.error || 'Generation failed'));
                        }
                    } catch (err) {
                        clearInterval(checkInterval);
                        reject(err);
                    }
                }, 3000);
                
                // Timeout after 10 minutes
                setTimeout(() => {
                    clearInterval(checkInterval);
                    reject(new Error('Generation timeout'));
                }, 600000);
                
            } catch (err) {
                reject(err);
            }
        });
    }
};

async function init() {
    try {
        // Load config
        state.config = await api.get('/api/config');
        
        // Populate dropdowns
        ui.populateSelect('promptType', state.config.prompt_types);
        ui.populateSelect('model', state.config.models);
        ui.populateSelect('voice', state.config.voices);
        ui.populateSelect('subtitleStyle', state.config.subtitle_styles);
        ui.populateSelect('subtitleAnimation', state.config.subtitle_animations);
        ui.populateSelect('filterPromptType', state.config.prompt_types, true);
        
        // Load preferences
        const prefs = JSON.parse(localStorage.getItem('videoGenPrefs') || '{}');
        if (prefs.formData) {
            for (const [k, v] of Object.entries(prefs.formData)) {
                const el = document.querySelector(`[name="${k}"]`);
                if (el) el.type === 'checkbox' ? el.checked = v : el.value = v;
            }
        }
        
        // Load queue
        queue.load();
        queue.render();
        queue.updateStats();
        
        // Setup event listeners
        $('generateForm').onsubmit = handleGenerate;
        $('generateForm').onchange = utils.debounce(() => {
            const data = Object.fromEntries(new FormData($('generateForm')));
            localStorage.setItem('videoGenPrefs', JSON.stringify({ formData: data }));
        }, 1000);
        
        $('addToQueueBtn').onclick = handleAddToQueue;
        $('startQueueBtn').onclick = () => queue.process();
        $('clearQueueBtn').onclick = async () => {
            if (await ui.confirm('Clear Queue', 'Are you sure you want to clear the entire queue?')) {
                queue.clear();
                ui.showNotification('Queue cleared', 'info');
            }
        };
        
        $('refreshGallery').onclick = () => loadVideos(true);
        $('filterPromptType').onchange = handleFilter;
        $('searchVideos').oninput = utils.debounce(handleSearch, 300);
        
        const modalClose = $$('.modal-close');
        if (modalClose) modalClose.onclick = closeModal;
        
        $('videoModal').onclick = e => e.target === $('videoModal') && closeModal();
        
        // Keyboard shortcuts
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') closeModal();
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !state.isGenerating) {
                $('generateForm').dispatchEvent(new Event('submit'));
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'q') {
                e.preventDefault();
                $('addToQueueBtn').click();
            }
        });
        
        // Load videos
        await loadVideos();
        
    } catch (err) {
        console.error('Init error:', err);
        ui.showNotification('Failed to initialize: ' + err.message, 'error');
    }
}

async function handleGenerate(e) {
    e.preventDefault();
    
    if (state.isGenerating || state.queueProcessing) {
        ui.showNotification('Generation already in progress', 'warning');
        return;
    }
    
    const formData = new FormData(e.target);
    const topic = formData.get('topic')?.trim();
    
    if (!topic || topic.length < 3) {
        ui.showNotification('Topic must be at least 3 characters', 'error');
        return;
    }
    
    const data = {
        topic,
        prompt_type: formData.get('promptType'),
        model: formData.get('model'),
        voice: formData.get('voice'),
        fps: parseInt(formData.get('fps')),
        aspect_ratio: [
            parseInt(formData.get('aspectRatio').split(':')[0]),
            parseInt(formData.get('aspectRatio').split(':')[1])
        ],
        transition_duration: parseFloat(formData.get('transitionDuration')),
        pan_effect: formData.get('panEffect') === 'on',
        zoom_effect: formData.get('zoomEffect') === 'on',
        subtitles: formData.get('subtitles') === 'on',
        subtitle_style: formData.get('subtitleStyle'),
        subtitle_animation: formData.get('subtitleAnimation'),
        highlight_keywords: formData.get('highlightKeywords') === 'on'
    };
    
    try {
        state.isGenerating = true;
        ui.setFormState(false);
        
        const result = await api.post('/api/generate', data);
        state.currentTaskId = result.task_id;
        
        $('progressSection').style.display = 'block';
        startProgressMonitoring();
        
    } catch (err) {
        ui.showNotification('Failed to start generation: ' + err.message, 'error');
        resetForm();
    }
}

function handleAddToQueue(e) {
    e.preventDefault();
    
    const formData = new FormData($('generateForm'));
    const topic = formData.get('topic')?.trim();
    
    if (!topic || topic.length < 3) {
        ui.showNotification('Topic must be at least 3 characters', 'error');
        return;
    }
    
    const data = {
        topic,
        prompt_type: formData.get('promptType'),
        model: formData.get('model'),
        voice: formData.get('voice'),
        fps: parseInt(formData.get('fps')),
        aspect_ratio: [
            parseInt(formData.get('aspectRatio').split(':')[0]),
            parseInt(formData.get('aspectRatio').split(':')[1])
        ],
        transition_duration: parseFloat(formData.get('transitionDuration')),
        pan_effect: formData.get('panEffect') === 'on',
        zoom_effect: formData.get('zoomEffect') === 'on',
        subtitles: formData.get('subtitles') === 'on',
        subtitle_style: formData.get('subtitleStyle'),
        subtitle_animation: formData.get('subtitleAnimation'),
        highlight_keywords: formData.get('highlightKeywords') === 'on'
    };
    
    queue.add(data);
    
    // Clear topic field
    $('topic').value = '';
    $('topic').focus();
}

function startProgressMonitoring() {
    let failCount = 0;
    
    state.progressInterval = setInterval(async () => {
        try {
            const status = await api.get(`/api/status/${state.currentTaskId}`);
            failCount = 0;
            
            ui.updateProgress(status.progress, status.stage);
            
            if (status.status === 'completed') {
                clearInterval(state.progressInterval);
                ui.updateProgress(100, '✓ Video generated successfully!');
                ui.showNotification('Video generated successfully!', 'success');
                
                setTimeout(() => {
                    resetForm();
                    loadVideos();
                }, 2000);
                
            } else if (status.status === 'failed') {
                clearInterval(state.progressInterval);
                ui.showNotification('Generation failed: ' + (status.error || 'Unknown error'), 'error');
                resetForm();
            }
            
        } catch (err) {
            if (++failCount >= 3) {
                clearInterval(state.progressInterval);
                ui.showNotification('Lost connection to server', 'error');
                resetForm();
            }
        }
    }, 2000);
}

function resetForm() {
    state.isGenerating = false;
    ui.setFormState(true);
    
    $('progressSection').style.display = 'none';
    $('progressFill').style.width = '0%';
    $('progressStage').textContent = 'Initializing...';
    
    state.currentTaskId = null;
    if (state.progressInterval) {
        clearInterval(state.progressInterval);
        state.progressInterval = null;
    }
}

async function loadVideos(refresh = false) {
    try {
        const data = await api.get('/api/videos');
        state.videos = data.videos || [];
        renderGallery();
        if (refresh) ui.showNotification('Gallery refreshed', 'success');
    } catch (err) {
        console.error('Load videos error:', err);
        ui.showNotification('Failed to load videos', 'error');
        state.videos = [];
        renderGallery();
    }
}

function handleSearch(e) {
    state.searchQuery = e.target.value.toLowerCase();
    state.currentPage = 1;
    renderGallery();
}

function handleFilter(e) {
    state.filterType = e.target.value;
    state.currentPage = 1;
    renderGallery();
}

function getFilteredVideos() {
    let filtered = state.videos;
    
    if (state.filterType) {
        filtered = filtered.filter(v => v.prompt_type === state.filterType);
    }
    
    if (state.searchQuery) {
        filtered = filtered.filter(v => 
            (v.topic || '').toLowerCase().includes(state.searchQuery) ||
            (v.model || '').toLowerCase().includes(state.searchQuery) ||
            (v.voice || '').toLowerCase().includes(state.searchQuery)
        );
    }
    
    return filtered;
}

function renderGallery() {
    const filtered = getFilteredVideos();
    const totalPages = Math.ceil(filtered.length / state.videosPerPage);
    
    if (state.currentPage > totalPages) state.currentPage = totalPages || 1;
    
    const start = (state.currentPage - 1) * state.videosPerPage;
    const pageVideos = filtered.slice(start, start + state.videosPerPage);
    
    const videoGallery = $('videoGallery');
    const emptyGallery = $('emptyGallery');
    
    if (pageVideos.length === 0) {
        if (videoGallery) videoGallery.style.display = 'none';
        if (emptyGallery) emptyGallery.style.display = 'block';
        $('paginationContainer').innerHTML = '';
        return;
    }
    
    if (videoGallery) videoGallery.style.display = 'grid';
    if (emptyGallery) emptyGallery.style.display = 'none';
    
    if (videoGallery) {
        videoGallery.innerHTML = pageVideos.map(createVideoCard).join('');
        
        $$$('.video-card', videoGallery).forEach((card, i) => {
            card.style.animationDelay = `${i * 50}ms`;
            card.classList.add('fade-in');
        });
    }
    
    renderPagination(totalPages);
}

function createVideoCard(video) {
    const duration = video.duration ? utils.formatDuration(video.duration) : '';
    
    return `
        <div class="video-card" data-id="${video.id}">
            <div class="video-thumbnail">
                <img src="/api/video/${video.id}/thumbnail" 
                     alt="${utils.escapeHtml(video.topic)}"
                     loading="lazy"
                     onerror="this.style.display='none'; this.parentElement.innerHTML='🎬'">
                ${duration ? `<span class="video-duration">${duration}</span>` : ''}
            </div>
            <div class="video-info">
                <div class="video-title" title="${utils.escapeHtml(video.topic)}">${utils.escapeHtml(video.topic)}</div>
                <div class="video-meta">
                    <span>${utils.formatText(video.prompt_type)}</span>
                    <span>📊 ${video.model} | 🎤 ${utils.formatText(video.voice)}</span>
                    <span>📅 ${utils.formatDate(video.created_at)}</span>
                    <span>💾 ${utils.formatSize(video.size)}</span>
                </div>
                <div class="video-actions">
                    <button class="play-btn" onclick="playVideo('${video.id}')">▶ Play</button>
                    <button class="download-btn" onclick="downloadVideo('${video.id}')">⬇ Download</button>
                    <button class="delete-btn" onclick="deleteVideo('${video.id}')">🗑</button>
                </div>
            </div>
        </div>
    `;
}

function renderPagination(totalPages) {
    const container = $('paginationContainer');
    if (!container) return;
    
    if (totalPages <= 1) {
        container.innerHTML = '';
        return;
    }
    
    const btns = [];
    
    btns.push(`<button class="pagination-btn" onclick="changePage(${state.currentPage - 1})" 
                ${state.currentPage === 1 ? 'disabled' : ''}>Previous</button>`);
    
    const maxVisible = 5;
    let start = Math.max(1, state.currentPage - Math.floor(maxVisible / 2));
    let end = Math.min(totalPages, start + maxVisible - 1);
    
    if (end - start < maxVisible - 1) {
        start = Math.max(1, end - maxVisible + 1);
    }
    
    if (start > 1) {
        btns.push(`<button class="pagination-btn" onclick="changePage(1)">1</button>`);
        if (start > 2) btns.push(`<span>...</span>`);
    }
    
    for (let i = start; i <= end; i++) {
        btns.push(`<button class="pagination-btn ${i === state.currentPage ? 'active' : ''}" 
                    onclick="changePage(${i})">${i}</button>`);
    }
    
    if (end < totalPages) {
        if (end < totalPages - 1) btns.push(`<span>...</span>`);
        btns.push(`<button class="pagination-btn" onclick="changePage(${totalPages})">${totalPages}</button>`);
    }
    
    btns.push(`<button class="pagination-btn" onclick="changePage(${state.currentPage + 1})" 
                ${state.currentPage === totalPages ? 'disabled' : ''}>Next</button>`);
    
    container.innerHTML = `<div class="pagination">${btns.join('')}</div>`;
}

// Global functions
window.changePage = page => {
    state.currentPage = page;
    renderGallery();
    const videoGallery = $('videoGallery');
    if (videoGallery) videoGallery.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

window.playVideo = async id => {
    try {
        const video = state.videos.find(v => v.id === id);
        if (!video) throw new Error('Video not found');
        
        const modal = $('videoModal');
        const modalVideo = $('modalVideo');
        const modalDetails = $('modalDetails');
        
        if (modal) modal.style.display = 'flex';
        if (modalVideo) modalVideo.src = `/api/video/${id}`;
        
        await new Promise((resolve, reject) => {
            if (modalVideo) {
                modalVideo.onloadedmetadata = resolve;
                modalVideo.onerror = reject;
            }
            setTimeout(() => reject(new Error('Load timeout')), 10000);
        });
        
        if (modalDetails) {
            modalDetails.innerHTML = `
                <h3>${utils.escapeHtml(video.topic)}</h3>
                <div class="video-details-grid">
                    <div class="detail-item">
                        <strong>Type</strong>
                        ${utils.formatText(video.prompt_type)}
                    </div>
                    <div class="detail-item">
                        <strong>Model</strong>
                        ${video.model}
                    </div>
                    <div class="detail-item">
                        <strong>Voice</strong>
                        ${utils.formatText(video.voice)}
                    </div>
                    <div class="detail-item">
                        <strong>Created</strong>
                        ${utils.formatDate(video.created_at, true)}
                    </div>
                    <div class="detail-item">
                        <strong>Size</strong>
                        ${utils.formatSize(video.size)}
                    </div>
                    <div class="detail-item">
                        <strong>Duration</strong>
                        ${utils.formatDuration(modalVideo.duration)}
                    </div>
                </div>
            `;
        }
        
        if (modalVideo) modalVideo.play().catch(() => {});
        
    } catch (err) {
        ui.showNotification('Failed to load video', 'error');
        closeModal();
    }
};

window.downloadVideo = id => {
    const video = state.videos.find(v => v.id === id);
    if (!video) return;
    
    const a = document.createElement('a');
    a.href = `/api/video/${id}`;
    a.download = `${(video.topic || 'video').replace(/[^a-z0-9]/gi, '_').toLowerCase()}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    ui.showNotification('Download started...', 'info');
};

window.deleteVideo = async id => {
    const video = state.videos.find(v => v.id === id);
    if (!video) return;
    
    const confirmed = await ui.confirm(
        'Delete Video',
        `Are you sure you want to delete "${video.topic}"? This cannot be undone.`
    );
    
    if (!confirmed) return;
    
    try {
        await api.delete(`/api/video/${id}`);
        state.videos = state.videos.filter(v => v.id !== id);
        renderGallery();
        ui.showNotification('Video deleted successfully', 'success');
        
        const modalVideo = $('modalVideo');
        if (modalVideo && modalVideo.src.includes(id)) closeModal();
        
    } catch (err) {
        ui.showNotification('Failed to delete video: ' + err.message, 'error');
    }
};

window.removeFromQueue = id => {
    queue.remove(id);
};

function closeModal() {
    const modal = $('videoModal');
    const modalVideo = $('modalVideo');
    const modalDetails = $('modalDetails');
    
    if (modal) modal.style.display = 'none';
    if (modalVideo) {
        modalVideo.pause();
        modalVideo.src = '';
    }
    if (modalDetails) modalDetails.innerHTML = '';
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);