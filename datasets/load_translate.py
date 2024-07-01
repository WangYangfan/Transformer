import datasets
from datasets import DownloadManager, DatasetInfo, load_dataset
import json

logger = datasets.logging.get_logger(__name__)

_EN = "en"
_ZH = "zh"

class Translate(datasets.GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description="英文到中文的翻译数据集",
            features=datasets.Features({
                _EN: datasets.Value("string"),
                _ZH: datasets.Value("string"),
            })
        )
    
    def _split_generators(self, dl_manager: DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'file_path': "./train.json",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    'file_path': "./test.json",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'file_path': "./dev.json",
                }
            )
        ]
    
    def _generate_examples(self, file_path):
        with open(file_path, 'r') as f:
            sentences = json.load(f)
        for idx, sample in enumerate(sentences):
            example = {_EN: sample[0].strip(), _ZH: sample[1].strip()}
            yield idx, example


if __name__ == '__main__':
    """
    DatasetDict({
        train: Dataset({
            features: ['en', 'zh'],
            num_rows: 176943
        })
        test: Dataset({
            features: ['en', 'zh'],
            num_rows: 50556
        })
        validation: Dataset({
            features: ['en', 'zh'],
            num_rows: 25278
        })
    })
    """

    dataset = load_dataset("./load_translate.py", trust_remote_code=True)
    # May report warning "Repo card metadata block was not found. Setting CardData to empty. HF google storage unreachable. Downloading and preparing it from source", but it doesn't matter.
    print(dataset)  

    dataset.save_to_disk("./translate")  # 保存构造的数据集，下次直接导入即可
    
