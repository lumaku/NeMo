# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, delete_extra_space, delete_space
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.en.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.taggers.word import WordFst
from nemo_text_processing.text_normalization.ru.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.ru.taggers.date import DateFst
from nemo_text_processing.text_normalization.ru.taggers.decimals import DecimalFst
from nemo_text_processing.text_normalization.ru.taggers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.ru.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.ru.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.ru.taggers.number_names import NumberNamesFst
from nemo_text_processing.text_normalization.ru.taggers.numbers_alternatives import AlternativeFormatsFst
from nemo_text_processing.text_normalization.ru.taggers.ordinal import OrdinalFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, input_case: str, deterministic: bool = True):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)
        print('Ru TN only support non-deterministic cases and produces multiple normalization options.')

        numbert_names = NumberNamesFst()
        alternative_formats = AlternativeFormatsFst()

        cardinal = CardinalFst(
            number_names=numbert_names, alternative_formats=alternative_formats, deterministic=deterministic
        )
        self.cardinal = cardinal
        cardinal_graph = cardinal.fst

        ordinal = OrdinalFst(
            number_names=numbert_names, alternative_formats=alternative_formats, deterministic=deterministic
        )
        self.ordinal = ordinal
        ordinal_graph = ordinal.fst

        self.decimal = DecimalFst(cardinal=cardinal, ordinal=ordinal, deterministic=deterministic)
        decimal_graph = self.decimal.fst

        # fraction = FractionFst(deterministic=deterministic, cardinal=cardinal)
        # fraction_graph = fraction.fst
        self.measure = MeasureFst(cardinal=cardinal, decimal=self.decimal, deterministic=deterministic)
        measure_graph = self.measure.fst
        self.date = DateFst(cardinal=cardinal, ordinal=ordinal, deterministic=deterministic)
        date_graph = self.date.fst
        word_graph = WordFst(deterministic=deterministic).fst
        # time_graph = TimeFst(cardinal=cardinal, deterministic=deterministic).fst
        # telephone_graph = TelephoneFst(deterministic=deterministic).fst
        self.electronic = ElectronicFst(deterministic=deterministic)
        electronic_graph = self.electronic.fst
        self.money = MoneyFst(cardinal=cardinal, decimal=self.decimal, deterministic=deterministic)
        money_graph = self.money.fst
        whitelist_graph = WhiteListFst(input_case=input_case, deterministic=deterministic).fst
        punct_graph = PunctuationFst(deterministic=deterministic).fst

        classify = (
            pynutil.add_weight(whitelist_graph, 1.01)
            # | pynutil.add_weight(time_graph, 1.1)
            | pynutil.add_weight(date_graph, 1.09)
            | pynutil.add_weight(decimal_graph, 1.1)
            | pynutil.add_weight(measure_graph, 1.1)
            | pynutil.add_weight(cardinal_graph, 1.1)
            | pynutil.add_weight(ordinal_graph, 1.1)
            | pynutil.add_weight(money_graph, 1.1)
            # | pynutil.add_weight(telephone_graph, 1.1)
            | pynutil.add_weight(electronic_graph, 1.1)
            # | pynutil.add_weight(fraction_graph, 1.1)
            | pynutil.add_weight(word_graph, 100)
        )

        # if not deterministic:
        #     roman_graph = RomanFst(deterministic=deterministic).fst
        #     universal_graph = UniversalFst(deterministic=deterministic).fst
        #     # the weight for roman_graph matches the word_graph weight for "I" cases in long sentences with multiple semiotic tokens
        #     classify |= pynutil.add_weight(roman_graph, 100) | pynutil.add_weight(universal_graph, 1.1)

        punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
        token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
        token_plus_punct = (
            pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
        )

        graph = token_plus_punct + pynini.closure(pynutil.add_weight(delete_extra_space, 1.1) + token_plus_punct)
        graph = delete_space + graph + delete_space

        self.fst = graph.optimize()