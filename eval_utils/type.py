from typing import Optional
from enum import Enum

class EvalErrorCode(Enum):
    string_above_max_length = "string_above_max_length"
    context_length_exceeded = "context_length_exceeded"
    output_parse_error      = "output_parse_error"
    other_error             = "other_error"

class EvalError(Exception):
    def __init__(self, code: EvalErrorCode, content: Optional[str] = None, inner_error: Optional[Exception] = None):
        if inner_error is not None:
            super().__init__(code, content, *inner_error.args)
        else:
            super().__init__(code, content)
        self.code = code
        self.content = content
        self.inner_error = inner_error

    def __repr__(self):
        args_joined = "\n".join(str(arg) for arg in self.args)
        return f"""EvalError(
        code        = {self.code}
        content     = {self.content}
        inner_error = (
            type={type(self.inner_error)}
            __str__={self.inner_error}
            args=(
            {args_joined}
        )
    )"""

class EvalInput:
    def __init__(self, model_with_platform: str, prompt: str):
        self.model_with_platform = model_with_platform  # e.g. "together/deepseek-ai/DeepSeek-V2"
        self.prompt = prompt

    @property
    def model_names(self):
        model_names = self.model_with_platform.split("_", maxsplit=1)
        assert len(model_names) == 2
        return model_names

    @property
    def model_platform(self):
        return self.model_names[0]

    @property
    def model_name(self):
        return self.model_names[1].replace("_", "/")

    @property
    def messages(self):
        return [{"role": "user", "content": self.prompt}]

    def __repr__(self):
        return (
            f"EvalInput(\n"
            f"  model = {self.model_with_platform},\n"
            f"  prompt = {self.prompt[:200]}...\n"
            f")"
        )

class EvalOutput:
    def __init__(self, content: Optional[str], pred_original: Optional[bool] = None, eval_error: Optional[EvalError] = None):
        self.content = content
        self.pred_original = pred_original
        self.pred_final = pred_original if pred_original is not None else None
        self.eval_error = eval_error

    def __repr__(self):
        return f"EvalOutput(pred={self.pred_original}, content={self.content}, error={self.eval_error})"

    def accuracy(self, truth_label: bool) -> float:
        return 1.0 if self.pred_final == truth_label else 0.0
