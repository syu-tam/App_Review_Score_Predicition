from transformers import pipeline
from tqdm import tqdm

# データフレーム内のレビューを翻訳するメイン関数
def translate_reviews(dataframe, config):
    """
    各レビューに対し、言語検出と翻訳を行い、結果をデータフレームに追加する。
    """
    device = config.base.device

    # 言語検出と翻訳のパイプラインを作成
    lang_detector = pipeline("text-classification", model=config.translate.language_detector_model, device=device)
    translator = pipeline("translation", model=config.translate.translate_model, device=device)

    tqdm.pandas()  # プログレスバーをpandasに適用

    # 各レビューに対して、言語検出と翻訳を実施
    dataframe['translated_review'] = dataframe['review'].progress_apply(
        lambda text: process_review(text, lang_detector, translator)
    )
    
    # 既存の 'language' 列を削除（不要であれば削除）
    dataframe.drop(columns=['language'], inplace=True, errors='ignore')

    return dataframe

# レビューの言語検出と翻訳を実行する補助関数
def process_review(text, lang_detector, translator):
    """
    指定されたレビューに対し、言語検出を行い、必要に応じて翻訳を行う。
    """
    language = detect_language(text, lang_detector)
    translated_text = translate_to_english(text, language, translator)
    return translated_text

# 言語検出を実行する関数
def detect_language(text, lang_detector):
    """
    レビューの言語を検出し、その言語コードを返す。
    """
    try:
        result = lang_detector(text)
        return result[0]['label']  # 言語ラベルを返す
    except Exception as e:
        print(f"Language detection error: {str(e)}")
        return "unknown"  # エラー発生時は "unknown" を返す

# 翻訳を実行する関数
def translate_to_english(text, src_lang, translator):
    """
    レビューが英語以外の場合、翻訳を行い、英語テキストを返す。
    """
    if src_lang == 'en' or src_lang == 'unknown':
        return text  # 英語や検出不能な場合はそのまま返す

    try:
        translated = translator(text, src_lang=src_lang, tgt_lang='en')
        return translated[0]['translation_text']
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return f"Error: {str(e)}"
