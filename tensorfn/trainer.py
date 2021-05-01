from pydantic import BaseModel


class Trainer:
    def __init__(self):
        self.__has_state_dict = {}
        self.__additional_obj = {}

    def register(self, name, value):
        self.__additional_obj[name] = value

    def __getattr__(self, name):
        if name not in self.__additional_obj:
            raise AttributeError(f"Cannot find attribute {name}")

        return self.__additional_obj[name]

    def __setattr__(self, name, value):
        if hasattr(value, "state_dict"):
            self.__has_state_dict[name] = value

        super().__setattr__(name, value)

    def state_dict(self):
        result = {}

        for k, v in self.__has_state_dict.items():
            result[k] = v.state_dict()

        for k, v in self.__additional_obj.items():
            if isinstance(v, BaseModel):
                v = v.dict()

            result[k] = v

        return result

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.__has_state_dict:
                self.__has_state_dict[k].load_state_dict(v)

            if k in self.__additional_obj:
                if isinstance(self.__additional_obj[k], BaseModel):
                    self.__additional_obj[k] = self.__additional_obj[k].parse_obj(v)

                else:
                    self.__additional_obj[k] = v
