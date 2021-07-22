var writer = writer || {};

writer.quill = null;

writer.KEY_TOPIC = 'SCIFI_WRITING_TOPIC';
writer.KEY_CHARACTERS = 'SCIFI_WRITING_CHARACTERS';
writer.KEY_CONTENTS = 'SCIFI_WRITING_CONTENTS';

writer.aiButtonPosition = { left: 0, top: 0 };

writer.showAIButton = function() {
    $('#ai-button').css({
        left: writer.aiButtonPosition.left,
        top: writer.aiButtonPosition.top
    }).show();
};

writer.onClickAIButton = function() {
    if (!$('#topic').val() || !$('#characters').val()) {
        $('#messageBoxBody').text('请先输入故事背景和角色列表');
        $('#messageBox').modal('show');
    } else {
        $('#ai-button').hide();
        $('#ai-wait').css({
            left: writer.aiButtonPosition.left,
            top: writer.aiButtonPosition.top
        }).show();

        // 模拟远程调用得到AI提示文本的逻辑：这里通常应先从 writer.quill 中
        // 获得当前位置的上下文信息，以及其他控件中的故事背景和人物列表，
        // 传送给服务端，等待服务端返回结果后，显示 ai-suggests 提示菜单。
        window.setTimeout(() => {
            $('#ai-wait').hide();
            writer.showAISuggests([
                'AI科幻写作建议的候选句子示例01。' +
                    '仅用于展示UI设计，' +
                    '在正式系统中会被真正的AI生成结果替代。',
                'AI科幻写作建议的候选句子示例02。' +
                    '仅用于展示UI设计，' +
                    '在正式系统中会被真正的AI生成结果替代。',
                'AI科幻写作建议的候选句子示例03。' +
                    '仅用于展示UI设计，' +
                    '在正式系统中会被真正的AI生成结果替代。'
            ]);
        }, 2000);
    }
};

writer.switchToHumanFormat = function() {
    writer.quill.format('color', '#222');
};

writer.showAISuggests = function(suggests) {
    $('#ai-suggests').empty();
    for (let i = 0; i < suggests.length; i++) {
        const suggest = suggests[i];
        $('<a class="dropdown-item">' + suggest + '</a>')
            .appendTo('#ai-suggests')
            .on('click', () => {
                writer.hideAISuggests();
                writer.quill.focus();
                writer.quill.insertText(
                    writer.quill.getSelection().index,
                    suggest,
                    {
                        'color': '#66c'
                    }
                );
                writer.switchToHumanFormat();
            });
        $('<div class="dropdown-divider"></div')
            .appendTo('#ai-suggests');
    }
    $('<a class="dropdown-item">取消</a>')
        .appendTo('#ai-suggests')
        .on('click', () => {
            writer.hideAISuggests();
            writer.quill.focus();
            writer.switchToHumanFormat();
        });

    let left = Math.abs(writer.aiButtonPosition.left - window.innerWidth) > 200 ?
        writer.aiButtonPosition.left : writer.aiButtonPosition.left - 200;
    $('#ai-suggests').css({
        left: left,
        top: writer.aiButtonPosition.top
    }).addClass('show').show();
};

writer.hideAISuggests = function() {
    $('#ai-suggests').removeClass('show').hide();
};

writer.saveContents = function() {
    window.localStorage.setItem(writer.KEY_TOPIC, $('#topic').val());
    window.localStorage.setItem(writer.KEY_CHARACTERS,
                                $('#characters').val());
    let delta = writer.quill.getContents();
    let json = JSON.stringify(delta, null, 2);
    window.localStorage.setItem(writer.KEY_CONTENTS, json);
};

writer.loadContents = function() {
    let topic = window.localStorage.getItem(writer.KEY_TOPIC);
    if (topic) $('#topic').val(topic);
    let characters = window.localStorage.getItem(writer.KEY_CHARACTERS);
    if (characters) $('#characters').val(characters);
    let contents = window.localStorage.getItem(writer.KEY_CONTENTS);
    if (contents) {
        let delta = JSON.parse(contents);
        writer.quill.setContents(delta);
    }
};

writer.init = function() {
    writer.quill = new Quill('#quill-editor', {
        theme: 'bubble',
        placeholder: '',
        modules: {
            toolbar: false
        }
    });

    writer.quill.on('editor-change', (event, ...args) => {
        if (event === 'selection-change') {
            let range = args[0];
            if (range === null)
                return;
            let rect = writer.quill.getBounds(range.index, range.length);
            writer.aiButtonPosition.left = rect.right + 8;
            writer.aiButtonPosition.top = rect.top - 2;
            writer.showAIButton();
        } else if (event === 'text-change') {
            writer.saveContents();
        }
    });

    $('#ai-button').on('click', writer.onClickAIButton);
    $('#topic').on('change', writer.saveContents);
    $('#characters').on('change', writer.saveContents);
    writer.loadContents();
    writer.quill.setSelection(writer.quill.getLength(), 0);
    writer.switchToHumanFormat();
};

window.addEventListener('DOMContentLoaded', () => {
    writer.init();
});
