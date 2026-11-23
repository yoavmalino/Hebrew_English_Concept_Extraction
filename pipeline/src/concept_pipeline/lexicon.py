"""Word lists for stage 2: function words, the academic stop list and the LaTeX operator lexicon.

Data only, no logic. Words are written without niqqud and before prefix removal, as they appear in
summaries. Changing a list changes stage-2 output, so re-run the stage-2 report after an edit.
"""

import re

# Words that never start or end a candidate phrase and are never a candidate alone. They may sit
# inside a phrase ("מרחב של מטריצות"). Both languages, because courses mix Hebrew and English.
FUNCTION_WORDS = frozenset(
    """
    של את על אל עם מן כי אם או גם לא אין יש כל זה זו זאת אלה אלו הזה הזו הזאת האלה אותו אותה
    אותם אותן הוא היא הם הן אני אתה אנחנו אנו אשר כאשר אז אזי לכן כלומר למשל כגון עבור בין לפי
    אחרי לפני תחת מעל מתחת ללא בלי רק אך אבל אולם אלא כמו כך כן איך מה מי מתי למה איפה היכן האם
    יותר פחות מאוד הרבה מעט כמה עוד שוב תמיד אף אחד אחת שני שתי שתיים שלושה שלוש כדי בגלל מכיוון
    לגבי ביחס ע"י בד"כ אם"ם וכו וכד' שלו שלה שלהם שלהן שלנו לו לה להם להן בו בה בהם בהן ממנו
    ממנה מהם עצמו עצמה עצמם כאן שם פה היה הייתה היו יהיה תהיה יהיו להיות הינו הינה הנה ניתן
    אפשר צריך צריכה נקבל נניח נסמן נגדיר נקרא נקראת יהי יהיו תהי קיים קיימת קיימים מתקיים
    מתקיימת גורר גוררת שווה שווים שוות בהינתן נתון נתונה נתונים כלשהו כלשהי כלשהם ידי עליו עליה
    עליהם אליו אליה ביותר כבר עדיין אחר אחרת אחרים אחרות כזה כזו כאלה שבו שבה שבהם שהוא שהיא
    שהם לכל בכל מכל וכל הכל שאינו שאינה אינו אינה אינם אינן אותה מאשר בעוד וגם ואז וכן ולכן ואם או"ם כנ"ל
    a an the of to in on at by for from with as is are was were be been being it its this that
    these those which who whom whose what when where why how and or not no nor but if then else
    so than such there here we us our you your they them their he she his her i me my do does did
    has have had can could may might must shall should will would also only each every all any
    some both either neither into onto over under between about after before via per
    """.split()
) | frozenset(
    # Added 2026-11-17 from the dev courses' top-ranked noise (never held-out): connectives,
    # abbreviations and verb forms that are never a concept and that stage 5 ranked high
    # ("מתוך" first in חישובית 2). Verb forms are listed as written and as the model's lemma.
    """
    מתוך לדוגמא לדוגמה כיצד מדוע בדיוק ממש הרי היות פי ע"פ צ"ל מש"ל כש ול יתכן ייתכן לבד בהתאם
    מסוים מסוימת מסוימים מסוימות מימין משמאל ללמוד למד נשתמש השתמש נזכור הוכחנו הוכיח ניקח
    לקח נשים ראה רשם הסתכל אמר בנה יסומן תסומן תתקבל מתקבלת התקבל מהווה מהווים היווה שמכיל
    ויהיו תהיינה תקרא תקיים יהא להסתמך לתקן לבצע ביצע להעביר להקצות לבסס כותבים נתעסק נעגל נתאר
    תיאר מצוא
    """.split()
)

# Unit labels: a heading or line opens with them, so alone they are never a concept. Only a
# definition pattern or an English gloss in parentheses keeps one (design stage 2, exception).
UNIT_LABEL_WORDS = frozenset(
    """
    הגדרה הגדרות משפט משפטים למה טענה טענות מסקנה מסקנות דוגמה דוגמא דוגמאות תרגיל
    תרגילים הוכחה הערה הערות תזכורת אבחנה הבחנה
    definition definitions theorem theorems lemma claim corollary example examples exercise
    exercises proof remark remarks note notes
    """.split()
)

# Structural words of academic writing ("תוצאות", "סיכום"): dropped alone unless there is evidence
# they are a concept: a definition pattern, an English gloss, or a kept phrase that contains them
# ("מרחב התוצאות" keeps "תוצאה" in a probability course). Design stage 2, stop list.
STRUCTURAL_WORDS = frozenset(
    """
    תוצאה תוצאות סיכום סיכומים מבוא נושא נושאים פרק פרקים הרצאה הרצאות תרגול תרגולים שיעור עמוד
    סעיף סעיפים שאלה שאלות פתרון פתרונות תשובה תשובות חלק חלקים שלב שלבים דבר דברים מקרה מקרים
    סוג סוגים דרך צורה אופן עובדה חשוב חשובה חשובות חשובים נוסף נוספת נוספים נוספות כללי כללית
    הבא הבאה הבאים הבאות הקודם הקודמת ראשון ראשונה שנייה שלישי עיקרי עיקרית רגיל רגילה פשוט פשוטה
    הקדמה הסבר הסברים קורס מצגת שבוע שבועות סימון שימוש ניסוח כתיבה תיאור התייחסות
    introduction summary chapter lecture lectures tutorial section page question questions answer
    solution solutions part step steps case cases result results conclusion overview thing things
    """.split()
)

# LaTeX operators that name a concept (design section 8, layer 1). A match is a weak mention of
# the concept (strength 0.2). Each entry: pattern on the formula's LaTeX, Hebrew name, English name.
OPERATORS = (
    (re.compile(r"\\det\b"), "דטרמיננטה", "determinant"),
    (re.compile(r"\\dim\b"), "מימד", "dimension"),
    (re.compile(r"\\ker\b"), "גרעין", "kernel"),
    (re.compile(r"\\lim\b|\\lim_"), "גבול", "limit"),
    (re.compile(r"\\int\b|\\iint\b|\\iiint\b"), "אינטגרל", "integral"),
    (re.compile(r"\\oint\b"), "אינטגרל קווי", "line integral"),
    (re.compile(r"\\sum\b"), "סכום", "sum"),
    (re.compile(r"\\prod\b"), "מכפלה", "product"),
    (re.compile(r"\\nabla\b"), "גרדיאנט", "gradient"),
    (re.compile(r"\\partial\b"), "נגזרת חלקית", "partial derivative"),
    (re.compile(r"\\sup\b"), "סופרמום", "supremum"),
    (re.compile(r"\\inf\b"), "אינפימום", "infimum"),
    (re.compile(r"\\log\b"), "לוגריתם", "logarithm"),
    (re.compile(r"\\ln\b"), "לוגריתם טבעי", "natural logarithm"),
    (re.compile(r"\\binom\b"), "מקדם בינומי", "binomial coefficient"),
    (re.compile(r"\\gcd\b"), "מחלק משותף מקסימלי", "greatest common divisor"),
    (re.compile(r"\\operatorname\{rank\}|\\mathrm\{rank\}|\brank\s*\("), "דרגה", "rank"),
    (re.compile(r"\\operatorname\{tr\}|\\mathrm\{tr\}|\\text\{tr\}"), "עקבה", "trace"),
    (re.compile(r"\\operatorname\{Var\}|\\mathrm\{Var\}|\bVar\s*\("), "שונות", "variance"),
    (re.compile(r"\\operatorname\{Cov\}|\\mathrm\{Cov\}|\bCov\s*\("), "שונות משותפת", "covariance"),
    (re.compile(r"\\mathbb\{E\}|\\operatorname\{E\}"), "תוחלת", "expected value"),
    (re.compile(r"\\operatorname\{span\}|\bSp\s*[\\{(]|\bspan\s*[\\{(]"), "פרישה", "span"),
    (re.compile(r"\\arg\\?max|\\operatorname\{argmax\}"), "ארגמקס", "argmax"),
)
