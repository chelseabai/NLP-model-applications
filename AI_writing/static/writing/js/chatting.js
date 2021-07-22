var chatting = chatting || {};

chatting.HUMAN_TEMPLATE = '<div class="row"><div class="col-2""';

chatting.KEY_AUTHOR = 'SCIFI_CHATTING_AUTHOR';
chatting.KEY_DIALOG = 'SCIFI_CHATTING_DIALOG';

// 控制按钮是否可用的逻辑。如果正在获取后台结果，则按钮不可用。
chatting.isGenerating = false;

chatting.scrollToEnd = function() {
    $('#dialog-panel').scrollTop($('#dialog-panel')[0].scrollHeight);
};

chatting.appendDialog = function(humanName, humanInput) {
    chatting.isGenerating = true;

    const HUMAN_TEMPLATE = `<div class="row chatting-human p-2">
<div class="col-2 col-md-1 chatting-human-name">${humanName}</div>
<div class="col-10 col-md-11 chatting-human-input">${humanInput}</div>
</div>`;
    $(HUMAN_TEMPLATE).appendTo('#dialog-container');

    const AI_TEMPLATE = `<div class="row chatting-ai p-2">
<div class="col-2 col-md-1 chatting-ai-name">AI</div>
<div class="col-10 col-md-11 chatting-ai-input">
  <div id="ai-wait" class="ml-3"></div>
</div>
</div>`;
    $(AI_TEMPLATE).appendTo('#dialog-container');

    chatting.scrollToEnd();

    // 模拟远程调用得到AI提示文本的逻辑：这里通常应先从 dialog-panel 中
    // 获得当前上下文信息，传送给服务端，等待服务端返回结果
    window.setTimeout(() => {
        let waitButton = $('#ai-wait');
        let parent = waitButton.parent();
        waitButton.remove();
        parent.text('AI科幻写作自动生成的句子对话示例。仅用于展示UI设计，' +
                    '在正式系统中会被真正的AI生成结果替代。');
        $('#human-input').val('');
        chatting.saveContents();
        chatting.isGenerating = false;
        chatting.scrollToEnd();
    }, 2000);
};

chatting.rollbackDialog = function() {
    // 撤销最后一行（AI的话）
    let aiRow = $("#dialog-container > div").last();
    aiRow.hide('swing', () => {
        aiRow.remove();
        // 撤销倒数第二行（人的话）
        let humanRow = $("#dialog-container > div").last();
        humanRow.hide('swing', () => {
            humanRow.remove();
            chatting.saveContents();
        });
    });
};

chatting.saveContents = function() {
    window.localStorage.setItem(chatting.KEY_DIALOG,
                                $('#dialog-container').html());
    window.localStorage.setItem(chatting.KEY_AUTHOR,
                                $('#human-name').val());
};

chatting.loadContents = function() {
    let dialog = window.localStorage.getItem(chatting.KEY_DIALOG);
    if (dialog)
        $('#dialog-container').html(dialog);
    let author = window.localStorage.getItem(chatting.KEY_AUTHOR);
    if (author)
        $('#human-name').val(author);
};

chatting.init = function() {
    $('#dialog-form').submit((e) => {
        if (chatting.isGenerating)
            return;
        let humanName = $('#human-name').val();
        let humanInput = $('#human-input').val();
        if (!humanInput)
            return;
        chatting.appendDialog(humanName, humanInput);
        e.preventDefault();
    });

    $('#rollback-button').on('click', () => {
        if (chatting.isGenerating)
            return;
        chatting.isGenerating = true;
        chatting.scrollToEnd();
        chatting.rollbackDialog();
        chatting.isGenerating = false;
    });

    $('#clear-button').on('click', () => {
        if (chatting.isGenerating)
            return;
        $('#dialog-container').empty();
        chatting.saveContents();
    });

    $('#human-name').on('change', chatting.saveContents);

    // chatting.loadContents();
    chatting.scrollToEnd();
};

window.addEventListener('DOMContentLoaded', () => {
    chatting.init();
});
