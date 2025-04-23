import unittest
import formulas
import numpy as np


class TestRentalCarCalculation(unittest.TestCase):
    def test_rental_car_calculation(self):
        PATH = "/Users/rohan/Downloads/Rental Car Model (6).xlsx"

        xl_model = formulas.ExcelModel().loads(PATH).finish()

        name_of_file = PATH.split("/")[-1]

        params = {
            "MODEL": {
                "price": "C11",
                "downPaymentPercentage": "C13",
                "interestRate": "C16",
                "loanTerm": "C17",
                "ratePerDay": "C23",
                "daysUsedPerMonth": "C24",
                "monthsUntilSale": "C28",
                "operatingExpenses": "C29",
            },
        }
        params_values = {
            "MODEL": {
                "price": 40000,
                "downPaymentPercentage": 0.05,
                "interestRate": 0.085,
                "loanTerm": 40,
                "ratePerDay": 150,
                "daysUsedPerMonth": 12,
                "monthsUntilSale": 55,
                "operatingExpenses": 400,
            },
        }

        inputs = {}
        for sheet_name, sheet_params in params.items():
            for param_name, param_cell in sheet_params.items():
                inputs[f"'[{name_of_file}]{sheet_name}'!{param_cell}"] = params_values[
                    sheet_name
                ][param_name]

        # Calculate with inputs properly formatted
        result = xl_model.calculate(inputs=inputs)
        # Access results directly from calculation result

        f68_value = result.get(f"'[{name_of_file}]MODEL'!F68")
        print(f68_value)

        # average monthly profit
        assert f68_value.value == np.array([[581.2262768650985]])


if __name__ == "__main__":
    unittest.main()
