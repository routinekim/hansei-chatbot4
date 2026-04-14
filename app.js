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
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 120초 후 취소 (충분한 폴백 시간 확보)

    let startTime = performance.now();
    try {
        const backendUrl = "/chat";
        console.log(`[DEBUG] API 호출 시도: ${backendUrl}`);

        const response = await fetch(backendUrl, {
            method: 'POST',
            mode: 'cors',
            credentials: 'omit',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/plain' // 스트리밍 텍스트 허용
            },
            body: JSON.stringify({ query: text, history: chatHistory }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const status = response.status;
            let errorMsg = `서버 응답 오류 (${status})`;
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || errorMsg);
        }

        // 스트리밍 데이터 읽기
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = "";
        bubble.innerHTML = ""; // 로딩 메시지 제거

        // marked.js 설정 (줄바꿈 허용)
        marked.setOptions({
            breaks: true,
            gfm: true
        });

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            fullAnswer += chunk;

            // 실시간 마크다운 렌더링
            bubble.innerHTML = marked.parse(fullAnswer);
            scrollToBottom();
        }

        const duration = (performance.now() - startTime) / 1000;
        console.log(`✅ [답변 완료] 소요 시간: ${duration.toFixed(2)}s`);

        // 대화 기록 저장
        chatHistory.push({ role: 'user', content: text });
        chatHistory.push({ role: 'assistant', content: fullAnswer });
        if (chatHistory.length > 10) chatHistory = chatHistory.slice(-10);

    } catch (error) {
        clearTimeout(timeoutId);
        let msgToShow = error.message;
        
        if (error.name === 'AbortError') {
            msgToShow = '서버 응답 시간이 너무 길어 요청이 중단되었습니다. 일시적인 네트워크 지연일 수 있으니 다시 시도해 주세요.';
        } else if (error.message === 'Failed to fetch') {
            msgToShow = '서버에 연결할 수 없습니다. 서버 상태를 확인해 주세요.';
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
