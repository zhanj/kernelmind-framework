from kernelmind.core import Point, Line

class Ask(Point):
    def load(self, memory): return "prompt"  # 返回一个值，确保 process 被调用
    def process(self, _): return input("Ask something: ")  # 现在会被调用
    def save(self, memory, _, out): memory["q"] = out; return "default"

class Answer(Point):
    def load(self, memory): return memory["q"]
    def process(self, q): return f"Mock answer to: {q}"
    def save(self, memory, _, out): print("Answer:", out)

if __name__ == "__main__":
    ask = Ask()
    answer = Answer()
    ask >> answer
    memory = {}
    line = Line(entry=ask)
    line.run(memory)
