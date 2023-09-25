from llm_rs import SessionConfig, GenerationConfig, Gpt2

class Chainer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Chainer, cls).__new__(cls)
            cls._instance.init_chainer()
        return cls._instance

    def init_chainer(self):
        self.stop_words = ['<EOL>', '<eol>', '<Eol>', 'pertanyaan :', 'Human', 'human', 'Pertanyaan', '\n']
        self.previous_qa = []

        session_config = SessionConfig(
            threads=4,
            context_length=1300,
            prefer_mmap=False
        )

        self.generation_config = GenerationConfig(
            top_p=0.44,
            top_k=1,
            temperature=0.22,
            max_new_tokens=120,
            repetition_penalty=1.13,
            stop_words=self.stop_words
        )

        self.model = Gpt2("2midguifSfFt5SbHJsxP.bin", session_config=session_config)

    def chain(self, user_input):
        if self.previous_qa:
            previous_question, previous_answer = self.previous_qa[-1]
        else:
            previous_question, previous_answer = "", ""

        template = f"saya bisa menjawab pertanyaan dengan masalah kesehatan.\nHai! Saya adalah chatbot yang akan menjawab pertanyaan seputar kesehatan. Saya adalah chatbot, bukan manusia.\nanda dapat menanyakan saya pertanyaan seputar kesehatan melalui kolom teks dibawah.\n\nPertanyaan saya:\n{previous_question}\n\nJawaban anda:\n{previous_answer}\n\nPertanyaan saya: {user_input}.\nJawaban anda :"

        result = self.model.generate(template, generation_config=self.generation_config)
        response = result.text.strip()

        self.previous_qa.append((user_input, response))

        if len(self.previous_qa) > 1:
            self.previous_qa.pop(0)

        return response


generator = Chainer()

def generate(text_input):
    result = generator.chain(text_input)
    return result