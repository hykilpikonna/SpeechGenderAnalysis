import warnings
from datetime import datetime
from pathlib import Path

from telegram import Update, Message
from telegram.ext import Updater, CallbackContext, Dispatcher, CommandHandler, MessageHandler, \
    Filters

from ina_main import *

warnings.filterwarnings("ignore")


def r(u: Update, msg: str, md=True):
    updater.bot.sendMessage(chat_id=u.effective_chat.id, text=msg,
                            parse_mode='Markdown' if md else None)


def cmd_start(u: Update, c: CallbackContext):
    r(u, '欢迎! 点下面的录音按钮就可以开始啦w')


def process_audio(message: Message):
    # Only when replying to voice or audio
    audio = message.audio or message.voice
    if not audio:
        return

    # Download audio file
    date = datetime.now().strftime('%Y-%m-%d %H-%M')
    try:
        downloader = bot.getFile(audio.file_id)
    except:
        downloader = bot.getFile(audio.file_id)
    file = Path(tmpdir).joinpath(f'{date} {message.from_user.name[1:]}.mp3')
    print(downloader, '->', file)
    downloader.download(file)

    # Segment file
    seg = Segmenter()
    result = process(seg, [str(file.absolute())]).results[0]

    # Null case
    print(result.frames)
    if len(result.frames) == 0:
        bot.send_message(message.chat_id, '分析失败, 大概是音量太小或者时长太短吧, 再试试w')
        return

    # Draw results
    with draw_result(str(file), result) as buf:
        f, m, o, pf = get_result_percentages(result)
        msg = f"分析结果: {f*100:.0f}% 🙋‍♀️ | {m*100:.0f}% 🙋‍♂️ | {o*100:.0f}% 🚫\n" \
              f"(结果仅供参考, 如果结果不是你想要的，那就是模型的问题，欢迎反馈)\n" \
              f"" \
              f"(因为这个模型基于法语数据, 和中文发音习惯有差异, 所以这个识别结果可能不准)"
        bot.send_photo(message.chat_id, photo=buf, caption=msg,
                       reply_to_message_id=message.message_id)


def cmd_analyze(u: Update, c: CallbackContext):
    reply = u.effective_message.reply_to_message

    # Parse command
    text = u.effective_message.text
    if not text:
        return
    cmd = text.lower().split()[0].strip()

    if cmd[0] not in '!/':
        return
    cmd = cmd[1:]

    if cmd not in ['analyze', 'analyze-raw']:
        return

    if cmd == 'analyze-raw':
        raw = True

    if u.effective_user.id == reply.from_user.id:
        process_audio(reply)
    else:
        r(u, '只有自己能分析自己的音频哦 👀')


def on_audio(u: Update, c: CallbackContext):
    process_audio(u.effective_message)


if __name__ == '__main__':
    tmpdir = Path('audio_tmp')
    tmpdir.mkdir(exist_ok=True, parents=True)

    # Find telegram token
    path = Path(os.path.abspath(__file__)).parent
    db_path = path.joinpath('voice-bot-db.json')
    if 'tg_token' in os.environ:
        tg_token = os.environ['tg_token']
    else:
        with open(path.joinpath('voice-bot-token.txt'), 'r', encoding='utf-8') as f:
            tg_token = f.read().strip()

    # Telegram login
    updater = Updater(token=tg_token, use_context=True)
    dispatcher: Dispatcher = updater.dispatcher
    bot = updater.bot

    dispatcher.add_handler(CommandHandler('start', cmd_start, filters=Filters.chat_type.private))
    dispatcher.add_handler(CommandHandler('analyze', cmd_analyze, filters=Filters.reply))
    dispatcher.add_handler(MessageHandler(Filters.reply, cmd_analyze))
    dispatcher.add_handler(MessageHandler(Filters.voice & Filters.chat_type.private, on_audio))
    dispatcher.add_handler(MessageHandler(Filters.audio & Filters.chat_type.private, on_audio))

    print('Starting bot...')
    updater.start_polling()
