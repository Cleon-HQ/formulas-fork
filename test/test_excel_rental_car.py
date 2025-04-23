import unittest
import formulas
import numpy as np


class TestRentalCarCalculation(unittest.TestCase):
    def test_rental_car_calculation(self):
        PATH = "test/test_files/rental_car.xlsx"

        xl_model = formulas.ExcelModel().loads(PATH).finish()

        name_of_file = PATH.split("/")[-1]
        sheet_name = f"'[{name_of_file}]MODEL'"

        params = {
            "downPaymentPercentage": "C13",
            "interestRate": "C16",
            "loanTerm": "C17",
            "ratePerDay": "C23",
            "daysUsedPerMonth": "C24",
            "monthsUntilSale": "C28",
            "operatingExpenses": "C29",
        }

        result = xl_model.calculate(
            inputs={
                f"{sheet_name}!C11": 40000,
                f"{sheet_name}!{params['downPaymentPercentage']}": 0.05,
                f"{sheet_name}!{params['interestRate']}": 0.085,
                f"{sheet_name}!{params['loanTerm']}": 40,
                f"{sheet_name}!{params['ratePerDay']}": 150,
                f"{sheet_name}!{params['daysUsedPerMonth']}": 12,
                f"{sheet_name}!{params['monthsUntilSale']}": 55,
                f"{sheet_name}!{params['operatingExpenses']}": 400,
            }
        )

        f68_value = result.get(f"{sheet_name}!F68")

        # average monthly profit
        assert f68_value.value == np.array([[581.2262768650985]])


if __name__ == "__main__":
    unittest.main()
