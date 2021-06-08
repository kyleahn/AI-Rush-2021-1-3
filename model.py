import timm
import torch.nn as nn

class FewShotClassifier(nn.Module):
    def __init__(self, pretrained):
        super(FewShotClassifier, self).__init__()
        self.model = timm.create_model(pretrained, pretrained=True, num_classes=2)
        try:
            in_features = self.model.classifier.in_features
        except:
            in_features = self.model.head.fc.in_features
        self.linear0 = nn.Linear(in_features, 12)
        self.linear1 = nn.Linear(in_features, 53)
        self.linear2 = nn.Linear(in_features, 42)
        nn.init.xavier_normal_(self.linear0.weight)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.model.reset_classifier(0)

    def forward(self, x):
        x = self.model(x)
        logits0 = self.linear0(x)
        logits1 = self.linear1(x)
        logits2 = self.linear2(x)
        return logits0, logits1, logits2
