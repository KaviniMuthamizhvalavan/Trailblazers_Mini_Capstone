/**
 * Corporate Learning Path Recommender — Frontend Logic
 * Handles chat, timeline rendering, session management, and API calls.
 */

// ── State ────────────────────────────────────────────────────────────────
const state = {
    sessionId: null,
    learnerId: '',
    turnNumber: 0,
    constraints: [],
    learningPath: [],
    isLoading: false,
};

// ── DOM Elements ──────────────────────────────────────────────────────────
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const learnerSelect = document.getElementById('learner-select');
const newSessionBtn = document.getElementById('new-session-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const turnIndicator = document.getElementById('turn-indicator');
const constraintCount = document.getElementById('constraint-count');

// Timeline elements
const timelineEmpty = document.getElementById('timeline-empty');
const timelineTracks = document.getElementById('timeline-tracks');
const timelineStats = document.getElementById('timeline-stats');
const statCourses = document.getElementById('stat-courses');
const statHours = document.getElementById('stat-hours');
const statWeeks = document.getElementById('stat-weeks');
const constraintsPanel = document.getElementById('constraints-panel');
const constraintsList = document.getElementById('constraints-list');

// ── API Base URL ──────────────────────────────────────────────────────────
const API_BASE = window.location.origin;

// ── Initialize ────────────────────────────────────────────────────────────
async function init() {
    await loadLearners();
    setupEventListeners();
    newSession();
}

function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    newSessionBtn.addEventListener('click', newSession);
    learnerSelect.addEventListener('change', (e) => {
        state.learnerId = e.target.value;
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });
}

// ── API Calls ─────────────────────────────────────────────────────────────
async function loadLearners() {
    try {
        const response = await fetch(`${API_BASE}/learners`);
        const data = await response.json();
        
        data.learners.forEach(learner => {
            const option = document.createElement('option');
            option.value = learner.learner_id;
            option.textContent = `${learner.name} (${learner.learner_id}) — ${learner.department}`;
            learnerSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load learners:', error);
    }
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || state.isLoading) return;

    // Add user message to chat
    addMessage(message, 'user');
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Show loading
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: state.sessionId,
                learner_id: state.learnerId,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Chat request failed');
        }

        const data = await response.json();

        // Update state
        state.sessionId = data.session_id;
        state.turnNumber = data.turn_number;
        state.constraints = data.constraints || [];
        state.learningPath = data.learning_path || [];

        // Add AI response to chat
        addMessage(data.response, 'ai');

        // Update timeline
        renderTimeline(state.learningPath);

        // Update constraints display
        renderConstraints(state.constraints);

        // Update indicators
        updateIndicators();

    } catch (error) {
        console.error('Chat error:', error);
        addMessage(`Sorry, an error occurred: ${error.message}. Please try again.`, 'ai');
    } finally {
        setLoading(false);
    }
}

// ── Chat UI ───────────────────────────────────────────────────────────────
function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    const avatarSvg = type === 'ai'
        ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5z"/>
            <path d="M2 17l10 5 10-5"/>
            <path d="M2 12l10 5 10-5"/>
           </svg>`
        : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
           </svg>`;

    // Format content (convert markdown-like formatting)
    const formattedContent = formatMessage(content);

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatarSvg}</div>
        <div class="message-content">${formattedContent}</div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMessage(text) {
    if (!text) return '<p>No response received.</p>';
    
    // Convert **bold** to <strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert *italic* to <em>
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert newlines to paragraphs
    const paragraphs = text.split('\n\n');
    if (paragraphs.length > 1) {
        return paragraphs.map(p => {
            const trimmed = p.trim();
            if (!trimmed) return '';
            // Check if it's a list
            if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
                const items = trimmed.split('\n').map(item => 
                    `<li>${item.replace(/^[-•]\s*/, '')}</li>`
                ).join('');
                return `<ul>${items}</ul>`;
            }
            return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
        }).join('');
    }
    
    return `<p>${text.replace(/\n/g, '<br>')}</p>`;
}

// ── Timeline Rendering ───────────────────────────────────────────────────
function renderTimeline(learningPath) {
    if (!learningPath || learningPath.length === 0) {
        timelineEmpty.style.display = 'flex';
        timelineTracks.style.display = 'none';
        statCourses.textContent = '0 courses';
        statHours.textContent = '0 hours';
        statWeeks.textContent = '0 weeks';
        return;
    }

    timelineEmpty.style.display = 'none';
    timelineTracks.style.display = 'flex';

    // Calculate stats
    const totalCourses = learningPath.length;
    const totalHours = learningPath.reduce((sum, c) => sum + (c.estimated_hours || 0), 0);
    const maxWeek = Math.max(...learningPath.map(c => c.week_end || 0), 0);

    statCourses.textContent = `${totalCourses} course${totalCourses !== 1 ? 's' : ''}`;
    statHours.textContent = `${totalHours} hours`;
    statWeeks.textContent = `${maxWeek} weeks`;

    // Group courses by week_start
    const weekGroups = {};
    learningPath.forEach((course, index) => {
        const weekStart = course.week_start || 1;
        if (!weekGroups[weekStart]) {
            weekGroups[weekStart] = [];
        }
        weekGroups[weekStart].push({ ...course, index });
    });

    // Render
    timelineTracks.innerHTML = '';
    const sortedWeeks = Object.keys(weekGroups).sort((a, b) => Number(a) - Number(b));

    sortedWeeks.forEach(weekStart => {
        const group = weekGroups[weekStart];
        const weekDiv = document.createElement('div');
        weekDiv.className = 'timeline-week-group';

        const weekEnd = Math.max(...group.map(c => c.week_end || weekStart));
        const weekLabel = weekStart === weekEnd 
            ? `Week ${weekStart}`
            : `Weeks ${weekStart}–${weekEnd}`;

        weekDiv.innerHTML = `<div class="week-label">${weekLabel}</div>`;

        group.forEach((course, i) => {
            const courseEl = createCourseElement(course, i);
            weekDiv.appendChild(courseEl);
        });

        timelineTracks.appendChild(weekDiv);
    });
}

function createCourseElement(course, animationIndex) {
    const div = document.createElement('div');
    div.className = 'timeline-course';
    div.style.animationDelay = `${animationIndex * 80}ms`;

    const track = course.track || 'Cloud';
    const difficulty = course.difficulty_level || 'intermediate';
    const weekRange = course.week_start === course.week_end
        ? `W${course.week_start}`
        : `W${course.week_start}–${course.week_end}`;

    div.innerHTML = `
        <div class="course-track-bar track-${track}"></div>
        <div class="course-info">
            <div class="course-title-row">
                <span class="course-title">${escapeHtml(course.title || 'Unknown Course')}</span>
                <span class="course-id">${escapeHtml(course.course_id || '')}</span>
            </div>
            <div class="course-meta">
                <span class="course-meta-item">
                    <span class="dot difficulty-${difficulty}"></span>
                    ${capitalize(difficulty)}
                </span>
                <span class="course-meta-item">📚 ${track}</span>
                <span class="course-meta-item">⏱ ${course.estimated_hours || 0}h</span>
                ${course.prerequisite_ids && course.prerequisite_ids.length > 0 
                    ? `<span class="course-meta-item">🔗 Prereqs: ${course.prerequisite_ids.join(', ')}</span>` 
                    : ''}
            </div>
            ${course.reason ? `<div class="course-reason">${escapeHtml(course.reason)}</div>` : ''}
        </div>
        <div class="course-week-badge">
            <div class="week-badge-content">
                <div class="week-range">${weekRange}</div>
                <div class="hours">${course.estimated_hours || 0}h</div>
            </div>
        </div>
    `;

    return div;
}

// ── Constraints Rendering ────────────────────────────────────────────────
function renderConstraints(constraints) {
    if (!constraints || constraints.length === 0) {
        constraintsPanel.style.display = 'none';
        return;
    }

    constraintsPanel.style.display = 'block';
    constraintsList.innerHTML = '';

    constraints.forEach(c => {
        const tag = document.createElement('span');
        tag.className = `constraint-tag ${c.superseded ? 'superseded' : 'active'}`;
        tag.innerHTML = `
            <span class="constraint-turn">T${c.turn}</span>
            ${escapeHtml(c.constraint)}
        `;
        constraintsList.appendChild(tag);
    });
}

// ── Session Management ───────────────────────────────────────────────────
function newSession() {
    state.sessionId = null;
    state.turnNumber = 0;
    state.constraints = [];
    state.learningPath = [];
    state.isLoading = false;

    // Clear chat (keep welcome message)
    const messages = chatMessages.querySelectorAll('.message:not(.welcome-message)');
    messages.forEach(m => m.remove());

    // Reset timeline
    renderTimeline([]);
    renderConstraints([]);
    updateIndicators();

    // Focus input
    chatInput.focus();
}

// ── Helpers ──────────────────────────────────────────────────────────────
function setLoading(isLoading) {
    state.isLoading = isLoading;
    loadingOverlay.style.display = isLoading ? 'flex' : 'none';
    sendBtn.disabled = isLoading;
    chatInput.disabled = isLoading;
}

function updateIndicators() {
    turnIndicator.textContent = `Turn ${state.turnNumber}`;
    const activeCount = state.constraints.filter(c => !c.superseded).length;
    constraintCount.textContent = `${activeCount} constraint${activeCount !== 1 ? 's' : ''}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function capitalize(str) {
    return str ? str.charAt(0).toUpperCase() + str.slice(1) : '';
}

// ── Boot ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
