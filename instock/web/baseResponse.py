import json


class BaseResponse:
    def __init__(self, code=0, data=None, error=None, message="ok"):
        self.code = code
        self.data = data or {}
        self.error = error
        self.message = message

    def to_dict(self):
        return {
            "code": self.code,
            "data": self.data,
            "error": self.error,
            "message": self.message
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)
