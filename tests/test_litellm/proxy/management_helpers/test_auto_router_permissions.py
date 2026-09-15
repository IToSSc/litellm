from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.management_helpers.auto_router_permissions import (
    authorize_member_auto_router_dependencies,
    authorize_member_auto_router_team,
    authorize_member_auto_router_write,
    validate_member_auto_router_config,
)
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo, updateDeployment, updateLiteLLMParams


@dataclass(frozen=True)
class _ReadTable:
    row: BaseModel | None = None

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> BaseModel | None:
        return self.row


@dataclass(frozen=True)
class _PermissionDb:
    litellm_teammembership: _ReadTable = _ReadTable()
    litellm_organizationtable: _ReadTable = _ReadTable()
    litellm_projecttable: _ReadTable = _ReadTable()


@dataclass(frozen=True)
class _Client:
    db: _PermissionDb = _PermissionDb()


def _team(**updates: object) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable.model_validate(
        {
            "team_id": "team-a",
            "models": ["allowed"],
            "members_with_roles": [Member(user_id="owner", role="user")],
            "team_member_permissions": ["/auto_router/manage"],
            **updates,
        }
    )


def _actor(**updates: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth.model_validate(
        {"user_id": "owner", "user_role": "internal_user", "models": ["allowed"], **updates}
    )


@pytest.fixture
def catalog() -> Router:
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake"}}
            for name in ("allowed", "other")
        ]
    )


@pytest.mark.parametrize(
    "actor_updates,team_updates,premium,allowed",
    [
        ({}, {}, True, True),
        ({"team_id": UI_TEAM_ID}, {}, True, True),
        ({"team_id": "team-a"}, {}, True, True),
        ({"user_role": LitellmUserRoles.TEAM}, {}, True, True),
        ({"team_id": "team-b"}, {}, True, False),
        ({"user_id": None}, {}, True, False),
        ({"user_id": ""}, {}, True, False),
        ({"user_id": "peer"}, {}, True, False),
        ({"user_role": LitellmUserRoles.INTERNAL_USER_VIEW_ONLY}, {}, True, False),
        ({"user_role": LitellmUserRoles.CUSTOMER}, {}, True, False),
        ({}, {"team_member_permissions": []}, True, False),
        ({}, {"team_member_permissions": None}, True, False),
        ({}, {"blocked": True}, True, False),
        ({}, {}, False, False),
    ],
)
def test_opt_in_requires_live_named_membership_and_write_role(
    actor_updates: Mapping[str, object], team_updates: Mapping[str, object], premium: bool, allowed: bool
) -> None:
    if allowed:
        authorize_member_auto_router_team(
            user_api_key_dict=_actor(**actor_updates), team=_team(**team_updates), premium_user=premium
        )
        return
    with pytest.raises(HTTPException) as denied:
        authorize_member_auto_router_team(
            user_api_key_dict=_actor(**actor_updates), team=_team(**team_updates), premium_user=premium
        )
    assert denied.value.status_code == 403


@pytest.mark.parametrize("placement", ["inline", "normalized"])
@pytest.mark.parametrize(
    "overrides", [{"api_base": "https://example.invalid"}, {"api_key": "fake"}, {"metadata": {}}, {"model": "other"}]
)
def test_all_tier_parameter_representations_reject_privileged_overrides(
    placement: str, overrides: Mapping[str, object]
) -> None:
    entry: Final = {"model_name": "allowed", "litellm_params": overrides}
    config: Final = (
        {"tiers": {"SIMPLE": [entry]}}
        if placement == "inline"
        else {"tiers": {"SIMPLE": ["allowed"]}, "tier_model_configs": {"SIMPLE": [entry]}}
    )
    with pytest.raises(HTTPException) as denied:
        validate_member_auto_router_config(config)
    assert denied.value.status_code == 400


def test_tier_config_is_normalized_and_unknown_router_extras_are_rejected() -> None:
    validated: Final = validate_member_auto_router_config(
        {"tiers": {"SIMPLE": [{"model_name": "allowed", "litellm_params": {"reasoning_effort": "low"}}]}}
    )
    assert validated.tiers == {"SIMPLE": ["allowed"]}
    assert validated.tier_model_configs["SIMPLE"][0].litellm_params == {"reasoning_effort": "low"}
    assert validate_member_auto_router_config(validated.model_dump()).tiers == validated.tiers
    with pytest.raises(HTTPException):
        validate_member_auto_router_config({"tiers": {"SIMPLE": "allowed"}, "api_base": "https://example.invalid"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ceiling", ["allowed", "team", "key", "member", "project", "org", "missing-org", "blocked-team"]
)
async def test_dependency_authorization_uses_each_destination_ceiling(catalog: Router, ceiling: str) -> None:
    team: Final = _team(
        models=["other"] if ceiling == "team" else ["allowed"],
        organization_id="org-a" if ceiling in ("org", "missing-org") else None,
        blocked=ceiling == "blocked-team",
    )
    actor: Final = _actor(
        models=["other"] if ceiling == "key" else ["allowed"],
        project_id="project-a" if ceiling == "project" else None,
    )
    client: Final = _Client(
        _PermissionDb(
            litellm_teammembership=_ReadTable(
                LiteLLM_TeamMembership(
                    user_id="owner",
                    team_id="team-a",
                    litellm_budget_table=LiteLLM_BudgetTable(allowed_models=["other"]),
                )
                if ceiling == "member"
                else None
            ),
            litellm_organizationtable=_ReadTable(
                LiteLLM_OrganizationTable(
                    organization_id="org-a",
                    budget_id="budget",
                    created_by="admin",
                    updated_by="admin",
                    models=["other"],
                )
                if ceiling == "org"
                else None
            ),
            litellm_projecttable=_ReadTable(
                LiteLLM_ProjectTable(project_id="project-a", team_id="team-a", models=["other"])
                if ceiling == "project"
                else None
            ),
        )
    )
    config: Final = validate_member_auto_router_config({"tiers": {"SIMPLE": "allowed"}})
    if ceiling == "allowed":
        await authorize_member_auto_router_dependencies(
            config=config,
            default_model=None,
            user_api_key_dict=actor,
            team=team,
            prisma_client=client,
            llm_router=catalog,
        )
        return
    with pytest.raises((HTTPException, ProxyException)):
        await authorize_member_auto_router_dependencies(
            config=config,
            default_model=None,
            user_api_key_dict=actor,
            team=team,
            prisma_client=client,
            llm_router=catalog,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("key_config", [
    {},
    {"timeout": 17},
    {"model_list": [{"model_name": "allowed", "litellm_params": {"model": "openai/gpt-4o-mini"}}]},
])
@pytest.mark.parametrize("key_model", ["allowed", "all-team-models", "other"])
async def test_key_configuration_does_not_override_member_router_model_grants(
    catalog: Router, key_config: Mapping[str, object], key_model: str
) -> None:
    actor: Final = _actor(models=[key_model], config=key_config)
    operation: Final = authorize_member_auto_router_dependencies(
        config=validate_member_auto_router_config({"tiers": {"SIMPLE": "allowed"}}),
        default_model=None, user_api_key_dict=actor, team=_team(), prisma_client=_Client(), llm_router=catalog,
    )
    if key_model == "other":
        with pytest.raises(ProxyException, match="not allowed to access model"):
            await operation
        return
    await operation
    assert actor.config == key_config
    assert actor.models == [key_model]


@pytest.mark.asyncio
async def test_update_checks_db_creator_and_decrypts_an_omitted_default(
    catalog: Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    monkeypatch.setenv("LITELLM_SALT_KEY", "member-router-test-salt")
    existing: Final = Deployment(
        model_name="model_name_team-a_uuid",
        litellm_params=LiteLLM_Params(
            model=encrypt_value_helper("auto_router/complexity_router"),
            complexity_router_config={"tiers": {"SIMPLE": "allowed"}},
            complexity_router_default_model=encrypt_value_helper("allowed"),
        ),
        model_info=ModelInfo(id="router-a", team_id="team-a", team_public_model_name="my-router", created_by="forged"),
        created_by="owner",
    )
    patch: Final = updateDeployment(
        litellm_params=updateLiteLLMParams(complexity_router_config={"tiers": {"SIMPLE": "allowed"}})
    )
    granted: Final = await authorize_member_auto_router_write(
        incoming=patch,
        existing=existing,
        user_api_key_dict=_actor(),
        team=_team(),
        premium_user=True,
        prisma_client=_Client(),
        llm_router=catalog,
    )
    assert granted.default_model == "allowed"
    assert granted.public_name == "my-router"
    assert granted.model_id == "router-a"
    with pytest.raises(HTTPException) as denied:
        await authorize_member_auto_router_write(
            incoming=patch,
            existing=existing.model_copy(update={"created_by": "peer"}),
            user_api_key_dict=_actor(),
            team=_team(),
            premium_user=True,
            prisma_client=_Client(),
            llm_router=catalog,
        )
    assert denied.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch_fields",
    [
        {"model_name": "renamed"},
        {"blocked": False},
        {"model_info": {"team_id": "other-team"}},
        {"model_info": {"member_auto_router": False}},
        {"litellm_params": {"model": "auto_router/quality_router"}},
        {"litellm_params": {"api_key": "fake"}},
    ],
)
async def test_member_update_rejects_administrative_fields(catalog: Router, patch_fields: Mapping[str, object]) -> None:
    existing: Final = Deployment(
        model_name="my-router",
        litellm_params=LiteLLM_Params(
            model="auto_router/complexity_router", complexity_router_config={"tiers": {"SIMPLE": "allowed"}}
        ),
        model_info=ModelInfo(id="router-a", team_id="team-a"),
        created_by="owner",
    )
    incoming: Final = updateDeployment.model_validate(
        {"litellm_params": {"complexity_router_config": {"tiers": {"SIMPLE": "allowed"}}}, **patch_fields}
    )
    with pytest.raises(HTTPException) as denied:
        await authorize_member_auto_router_write(
            incoming=incoming,
            existing=existing,
            user_api_key_dict=_actor(),
            team=_team(),
            premium_user=True,
            prisma_client=_Client(),
            llm_router=catalog,
        )
    assert denied.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["missing", "nested"])
async def test_member_dependencies_require_plain_configured_models(target: str) -> None:
    catalog: Final = Router(
        model_list=[
            {"model_name": "allowed", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake"}},
            {
                "model_name": "nested",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "allowed"}},
                },
            },
        ]
    )
    with pytest.raises(HTTPException) as denied:
        await authorize_member_auto_router_dependencies(
            config=validate_member_auto_router_config({"tiers": {"SIMPLE": target}}),
            default_model=None,
            user_api_key_dict=_actor(models=[target]),
            team=_team(models=[target]),
            prisma_client=_Client(),
            llm_router=catalog,
        )
    assert denied.value.status_code == 400
