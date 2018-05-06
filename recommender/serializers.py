from django.contrib.auth.models import User, Group
from rest_framework import serializers
from recommender.models import Profile
from rest_framework.validators import UniqueValidator

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')

class RegistrationSerializer(serializers.HyperlinkedModelSerializer):
    """
    Convert User models to Serializable objects for registration
    """
    email = serializers.EmailField(label='Email', validators=[UniqueValidator(queryset=User.objects.all())])
    class Meta:
        model = User
        fields = ('username', 'email', 'password')
        extra_kwargs = {'password': {'write_only': True,
                                     'style': {'input_type': 'password'}}}

    def create(self, validated_data):
        user = User.objects.create(username=validated_data['username'], email=validated_data['email'])
        user.set_password(validated_data['password'])
        user.save()
        return user


class ProfileSerializer(serializers.HyperlinkedModelSerializer):
    user = UserSerializer() 
    class Meta:
        model = Profile
        fields = ('user', 'crawled')