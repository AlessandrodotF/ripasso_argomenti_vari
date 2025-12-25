import pandas as pd


address = pd.DataFrame(
    {
        "addressId": ["1", "2"],
        "personId": ["2", "3"],
        "city": ["Cassino", "Padova"],
        "state": ["Lazio", "Veneto"],
    }
)


person = pd.DataFrame(
    {
        "personId": ["2", "3"],
        "lastName": ["A", "F"],
        "firstName": ["D", "F"],
    }
)


# merge per esempio su personId
a = address.merge(person, on="personId", how="left")
print(a)
# swapp di colonne
a = a[["firstName", "lastName", "city"]]
print(a)

a.rename
## MySql
# SELECT firstName,lastName,city,state FROM
# Person
# left join Address on Person.personId = address.personId

# PostgreSQL
# SELECT firstName,lastName,city,state FROM
# Person
# left join Address on Person.personId = address.personId


# altro esercizio
def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    a = customers.merge(orders, left_on="id", right_on="customerId", how="left")
    a = a[a.isnull().any(axis=1) == True]["name"]
    new_a = pd.DataFrame(a)
    new_a.rename(columns={"name": "Customers"}, inplace=True)
    return new_a
