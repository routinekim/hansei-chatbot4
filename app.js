// 챗봇 이전 대화 기억(Context Memory) (최대 8개)
let chatHistory = [];

function handleEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function sendQuickMessage(text) {
    appendUserMessage(text);
    fetchChatResponse(text);
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();

    if (text === '') return;

    appendUserMessage(text);
    input.value = '';

    fetchChatResponse(text);
}

async function fetchChatResponse(text) {
    // 로딩 말풍선 표시
    const chatbox = document.getElementById('chatbox');
    const row = document.createElement('div');
    row.className = 'message-row bot-row';
    const avatar = document.createElement('div');
    avatar.className = 'bot-avatar';
    const img = document.createElement('img');
    img.src = 'hanbi.gif';
    img.alt = '한비';
    avatar.appendChild(img);

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble bot-bubble';
    bubble.innerHTML = '자료를 검토 중입니다... 잠시만 기다려주세요 👀<br>';

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatbox.appendChild(row);
    scrollToBottom();

    // 서버에 요청 전송 (타임아웃 90초 설정)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000); // 90초 후 취소

    try {
        const backendUrl = "https://hansei-chatbot4.onrender.com/api/chat";
        console.log(`[DEBUG] API 호출 시도: ${backendUrl}`);

        const response = await fetch(backendUrl, {
            method: 'POST',
            mode: 'cors', // Cross-origin 요청 명시
            credentials: 'omit', // 쿠키 등 인증 정보 제외 (보안/차단 방지)
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ query: text, history: chatHistory }),
            signal: controller.signal // 신호 전달
        });

        clearTimeout(timeoutId); // 요청 성공 시 타이머 해제

        if (!response.ok) {
            const status = response.status;
            let errorMsg = `서버 응답 오류 (${status})`;
            console.error(`[DEBUG] 서버 응답 코드: ${status} - URL: ${backendUrl}`);
            try {
                const errorData = await response.json();
                errorMsg = errorData.detail || errorMsg;
            } catch (e) {
                const text = await response.text().catch(() => "");
                if (text) errorMsg += `: ${text.substring(0, 80)}...`;
                console.error(`[DEBUG] 서버 응답 본문 파싱 실패 또는 텍스트: ${text}`);
            }
            throw new Error(errorMsg);
        }
        const data = await response.json();

        // 대화 기록 저장 (어시스턴트 응답과 함께)
        chatHistory.push({ role: 'user', content: text });
        chatHistory.push({ role: 'assistant', content: data.answer });
        // 최근 8개 대화만 유지
        if (chatHistory.length > 8) {
            chatHistory = chatHistory.slice(-8);
        }

        // marked.js 설정 (줄바꿈 허용)
        marked.setOptions({
            breaks: true,
            gfm: true
        });

        // Markdown 렌더링으로 말풍선 갱신
        bubble.innerHTML = marked.parse(data.answer);
    } catch (error) {
        clearTimeout(timeoutId);
        let msgToShow = error.message;
        
        if (error.name === 'AbortError') {
            msgToShow = '서버 응답 시간이 너무 길어 요청이 취소되었습니다. Render 무료 서버가 깨어나는 중일 수 있으니, 잠시 후 다시 시도해 주세요.';
        } else if (error.message === 'Failed to fetch') {
            msgToShow = '서버에 연결할 수 없습니다. 서버가 켜져 있는지 확인해 주세요.';
        }
        
        bubble.innerHTML = `⚠️ 질문 처리 실패<br><span style="font-size:0.8rem; color:#666;">${msgToShow}<br>🙏 계속해서 응답이 없다면 페이지를 새로고침(F5) 해주세요.</span>`;
        console.error('Chat Error:', error);
    }
    scrollToBottom();
}

function appendUserMessage(text) {
    const chatbox = document.getElementById('chatbox');

    const row = document.createElement('div');
    row.className = 'message-row user-row';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble user-bubble';
    bubble.textContent = text;

    row.appendChild(bubble);
    chatbox.appendChild(row);

    scrollToBottom();
}

function appendBotMessage(text) {
    const chatbox = document.getElementById('chatbox');

    const row = document.createElement('div');
    row.className = 'message-row bot-row';

    const avatar = document.createElement('div');
    avatar.className = 'bot-avatar';

    const img = document.createElement('img');
    img.src = 'hanbi.gif';
    img.alt = '한비';
    avatar.appendChild(img);

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble bot-bubble';
    bubble.textContent = text;

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatbox.appendChild(row);

    scrollToBottom();
}

function scrollToBottom() {
    const chatbox = document.getElementById('chatbox');
    chatbox.scrollTop = chatbox.scrollHeight;
}
