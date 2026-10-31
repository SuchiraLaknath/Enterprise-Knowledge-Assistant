import pytest

from api.auth import DEMO_USERS as API_USERS
from ui.demo_users import DEMO_USERS, format_user_label


@pytest.mark.parametrize(
    ("username", "label"),
    [("vimal", "vimal - viewer"), ("nadee", "nadee - analyst"), ("admin", "admin - admin")],
)
def test_a_user_label_shows_the_name_and_the_role(username, label):
    assert format_user_label(username) == label


def test_the_screen_shows_the_roles_and_passwords_the_api_really_uses():
    # A label that disagreed with the API would tell an evaluator the wrong story about access.
    assert set(DEMO_USERS) == set(API_USERS)
    for username, user in DEMO_USERS.items():
        assert user.role == API_USERS[username].role
        assert user.password == API_USERS[username].password
